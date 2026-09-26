# Plan V8 — ternarisation de qualité avec sélection discrète

> Historique, remplacé par le [plan V9](TERNARY_QUALITY_RECOVERY_PLAN_V9.md).
> Le [registre V9](TERNARY_V9_EVIDENCE_2026-09-25.md) corrige notamment
> l'interprétation des symlinks, la preuve de held-out, la couverture des
> trajectoires et la distinction TTQ/ternaire strict. Le contenu ci-dessous
> conserve les décisions et hypothèses d'alors, pas des résultats acquis.

Date : 25 septembre 2026  
Objectif : obtenir un DiT Stable Audio 3 réellement ternaire et utilisable,
ou démontrer proprement que la cible qualité/format est incompatible avec le
budget actuel.  
Contraintes : codes `{-1,0,+1}`, rechargement exact, ≤12 GB Metal pendant
l’entraînement, taille finale publiée et vérifiée, aucune cascade avant le
passage d’un bloc.

Ce plan remplace V7. Il ne promet pas qu’une méthode non testée réussira ; il
empêche surtout de dépenser du temps sur une QAT qui ne change pas les codes,
sur un corpus non indépendant ou sur un dernier checkpoint dégradé.

## 0. Définition de réussite

Un résultat ne sera appelé « modèle ternaire de qualité » que s’il satisfait
tous les points suivants :

1. chaque matrice du scope annoncé possède des codes exacts `{-1,0,+1}` ;
2. le forward d’entraînement, le packer, le reload cross-process et l’inférence
   utilisent le même record, sans re-quantification silencieuse ;
3. le modèle rechargé passe le train, une validation indépendante et un test
   réservé, avec mêmes prompts, seeds, sigmas et bruit ;
4. les gates de velocity, d’état terminal et d’audio brut passent ensemble ;
5. le scope complet et la taille réelle sur disque sont publiés. « 100 % » ne
   signifie pas seulement les 7 projections du bloc 0.

Gates de travail pour chaque bloc : velocity validation mean ≥`0,94`, min
≥`0,82`; terminal mean ≥`0,90`, min ≥`0,80`; aucun prompt validation ne doit
perdre plus de `0,03` par rapport au checkpoint source. Gate de release final :
velocity mean ≥`0,95`, min ≥`0,85`; terminal mean ≥`0,90`, min ≥`0,80`; six
audios bruts et test réservé conformes. Ces seuils sont des décisions de
projet, pas des résultats acquis.

## 1. P0 — rendre l’expérience falsifiable

### Livrables

- `dataset_contract.json` avec train/validation/test disjoints par source,
  prompt, latent et hash ;
- manifest unique pour 8 sigmas de production, 2 seeds minimum et les
  réinjections du sampler ;
- digest du code du trainer, du runtime MLX, du teacher, du quantizer et des
  caches ;
- validation indépendante jamais utilisée pour gradients, calibration de
  seuil ou sélection manuelle ;
- audit qui refuse un artifact dont les digests ne correspondent pas.

### Vérifications

1. teacher dense déterministe sur 16 états golden ;
2. mêmes sorties après sauvegarde/reload du teacher ;
3. record ternaire synthétique avec les trois codes et tous les champs ;
4. reload cross-process bit/float exact dans la tolérance contractuelle ;
5. mesure Metal active/réservée, RSS, swap, dtype et présence teacher/student.

**Stop P0 :** un seul mismatch de provenance, timestep, biais ou forward.

## 2. P1 — calibration d’activations time-aware

Le papier audio récent [Post-Training Quantization for Audio Diffusion
Transformers](https://arxiv.org/abs/2510.00313) montre que la distribution des
activations dépend du timestep. On enregistre donc, pour chaque projection
candidate, les entrées et sorties teacher sur les 8 sigmas, plusieurs prompts
et deux seeds. On calcule par canal : moyenne, écart-type, quantiles, maximum
robuste et corrélation avec le sigma.

Tester séparément :

- aucun lissage, contrôle historique ;
- lissage/normalisation par canal dépendant du timestep ;
- rotation Hadamard uniquement dans la représentation d’activation, avec
  inverse exact dans le kernel ;
- groupes G16/G32/G64 et partage ou non des paramètres selon la région temporelle.

HadaNorm/SpinQuant ne seront pas utilisés comme justification d’un poids
ternaire : ils restent des ablations d’activation tant que le forward de poids
ne respecte pas exactement le contrat.

**Gate P1 :** sur un bloc non entraîné, le meilleur candidat doit battre le
contrôle sur le minimum validation et le terminal, pas uniquement sur MSE des
poids. Sinon revenir au contrat ou réduire le scope ; ne pas lancer QAT.

## 3. P2 — solveur de codes discret depuis le master dense

Le cœur V8 change ici. Le master dense original est conservé ; les codes ne
sont pas appris uniquement par STE.

### Paramétrisation

Pour chaque groupe :

`W_hat = s_pos * 1[q=+1] - s_neg * 1[q=-1] + b` avec `q ∈ {-1,0,+1}`.

Le biais `b`, s’il est autorisé par le scope, est explicitement packé et
rechargé. Les niveaux et seuils ont des contraintes positives et des bornes
de trust-region autour de la projection initiale.

### Recherche en deux temps

1. **Proposition continue courte** : optimiser un surrogate soft avec sortie de
   bloc et velocity, mais conserver le master dense et les activations calibrées.
2. **Décision exacte** : pour chaque groupe ou tuile, tester les flips ternaires
   candidats par gain d’objectif ; accepter seulement une modification qui
   améliore le score validation ou le score train sans dépasser la pénalité de
   pire prompt. Recalculer exactement `W_hat`, reserialiser et recharger avant
   de valider.

Le solveur doit journaliser `codes_before/after`, distance au seuil, fraction
zéro, `s_pos/s_neg`, sorties de bloc et score par sigma. Une loss qui baisse
sans flip n’est pas un progrès. Un flip qui améliore seulement le train est
rejeté s’il dégrade validation.

Objectif initial, normalisé par batch :

`0,35 L_block_output + 0,35 L_velocity + 0,20 L_direction + 0,10 L_norm`.

Le rollout 4/8 pas n’entre qu’en validation de sélection au départ. Il ne peut
pas compenser une projection locale mauvaise.

**Gate P2 :** un bloc doit dépasser le candidat V7 sur `min_velocity`,
`terminal_min` et la moyenne validation après reload. Sinon arrêter cette
branche, essayer un autre group size ou conclure que ce bloc ne passe pas en
ternaire strict.

### Résultat P2 initial et correction obligatoire

Le premier solveur activation-aware V8 a été exécuté sur les 7 projections du
bloc 0. Il a réduit une MSE de poids pondérée, mais a donné velocity
`0,86667 / 0,49690` sur le benchmark train sélectionné, contre
`0,96946 / 0,89541` pour le record source. Sur validation, il tombe à
`0,84760 / 0,66785` et le terminal à `0,44607 / 0,28554`. Cette branche est
rejetée.

La suite P2 ne doit donc pas optimiser une approximation locale de `W`. Elle
doit utiliser un budget de propositions très limité et une acceptation par
sortie de bloc :

1. ouvrir une seule matrice et un seul groupe à la fois ;
2. générer au maximum quatre alternatives `(q, s+, s-)`, plus l'état courant ;
3. exécuter le bloc teacher/student sur des activations enregistrées pour les
   8 sigmas ;
4. mesurer `L_block_output`, velocity et le pire sigma sur train ;
5. vérifier le même candidat sur un mini-held-out validation non utilisé pour
   choisir le seuil ;
6. accepter seulement si mean, minimum et terminal proxy ne régressent pas ;
7. reserialiser et recharger après chaque lot d'acceptations.

La MSE de poids et le RMS d'activation deviennent des heuristiques de tri,
jamais un critère de promotion. Si un lot de flips ne passe pas la sortie du
bloc, il est annulé même si ses statistiques locales s'améliorent.

Le premier banc bloc→full-DiT confirme que le gate local est nécessaire mais
insuffisant : un swap `ff.2` atteint `0,98217/0,96097` mean/min au bloc, puis
`0,95741/0,77588` sur le DiT complet. La sélection finale doit donc faire
`block filter → full-DiT train → validation held-out → terminal`; aucune étape
ne peut être supprimée pour économiser du temps.

## 4. P3 — sensibilité et composition sûre

Les swaps V7 montrent que les modules ne sont pas équivalents. Mesurer pour
les 7 projections des blocs 0–1 :

- perte locale sur activation teacher ;
- perte de velocity par sigma ;
- perte terminale ;
- variation RMS/DC ;
- nombre de flips et coût mémoire.

Construire les compositions par rechargement du modèle complet, jamais par
addition des gains individuels. Sélectionner un ensemble Pareto mean/min/
terminal/min-terminal. Conserver un trust-region sur le candidat V7 afin
qu’une composition ne détruise pas le meilleur pire cas.

Les méthodes de sélection sélective de [QuEST](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_QuEST_Low-bit_Diffusion_Model_Quantization_via_Efficient_Selective_Finetuning_ICCV2025_paper.html)
servent de principe d’allocation, pas de preuve audio directe.

**Gate P3 :** pas plus de 7 matrices ouvertes à la fois ; aucune composition
ne passe si elle n’est pas reserialisée et auditée avec les autres matrices.

## 5. P4 — checkpoints et validation automatique

Modifier le trainer pour auditer toutes les 8–16 updates et conserver :

- `best_validation_mean` ;
- `best_validation_min` ;
- `best_terminal_min` ;
- le meilleur point Pareto ;
- le checkpoint initial comme référence immuable.

L’export choisit un checkpoint validé, jamais `latest`. Deux régressions
successives arrêtent la branche. La validation est séparée des exemples ayant
servi à choisir les seuils et les flips.

Le rollout devient une mesure de robustesse : 4 pas d’abord, 8 pas ensuite,
avec mêmes bruit et réinjections que le sampler de production. La pondération
par sigma est apprise uniquement sur train et enregistrée dans le manifest.

## 6. P5 — cascade bloc par bloc

Seulement après le passage P2/P3 sur le bloc 0 :

1. charger les blocs acceptés en hard packed ;
2. ouvrir un seul bloc suivant depuis son master dense ;
3. calibrer ses activations sur les états teacher et student réellement vus ;
4. résoudre les codes avec P2 ;
5. recharger, auditer et figer le meilleur checkpoint ;
6. comparer à une validation qui inclut les blocs déjà ternarisés.

Une régression rouvre le dernier bloc ; elle n’est jamais masquée par une
moyenne sur 24 blocs. Le test réservé reste fermé jusqu’à la fin du scope.

## 7. P6 — scope, taille et déploiement

Avant de prétendre « 100 % ternaire », produire une matrice de couverture :
24 blocs × toutes les matrices éligibles, paramètres continus exclus avec
justification. Compter réellement : codes packés, scales, seuils, biais,
padding, index, buffers FP16 et conteneur.

Le budget `455,8 MB` est une hypothèse à vérifier, pas une promesse. Le
release final doit être ≤`500 000 000` octets sur disque, afficher le nombre
de paramètres et réussir le reload indépendant. Si le scope complet ne tient
pas, publier la comptabilité et arrêter la revendication 100 %.

## 8. Ordre d’exécution concret

1. Implémenter P0 et la validation indépendante ; aucune QAT.
2. Ajouter l’enregistrement d’activation time-aware et le banc P1.
3. Implémenter le solveur discret P2 sur une seule projection synthétique,
   puis sur une seule matrice réelle avec scoring de sortie de bloc.
4. Tester sur `ff.2` puis `self_attn.to_qkv` du bloc 0, car les swaps V7
   indiquent une sensibilité mesurable.
5. Auditer chaque proposition rechargée et conserver le meilleur Pareto.
6. Étendre aux 7 projections seulement si le bloc passe P2.
7. Lancer P5 pour un bloc suivant, un seul à la fois.
8. Fermer avec P6, rendu audio brut et test réservé.

## 9. Conditions d’arrêt et fallback honnête

Après au plus trois familles de solveurs discrets, si le bloc 0 ne passe pas
les gates validation malgré une couverture time-aware et un master dense intact,
arrêter la prétention de « ternarisation qualité » pour ce DiT. Tester alors un
Pareto explicitement non strict : poids 4-bit sur les projections sensibles,
résidu low-rank ou précision mixte. Cette branche ne pourra pas être nommée
100 % ternaire, mais elle permettra de distinguer une impossibilité de budget
d’une erreur d’implémentation.

Ce fallback n’est pas lancé automatiquement : il est seulement la sortie
scientifiquement correcte si le gate strict échoue encore.
