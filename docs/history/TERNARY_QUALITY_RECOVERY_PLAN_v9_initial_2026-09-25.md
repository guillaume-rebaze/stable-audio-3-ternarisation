> Archive historique de la V9 initiale du 25 septembre 2026.
> Remplacée par la [V9 révisée — périmètre Bonsai](../TERNARY_QUALITY_RECOVERY_PLAN_V9.md).
> L'exigence toutes-matrices et le chiffre 10–30 % ci-dessous ne sont plus actifs.

# Plan V9 — réussir la conversion ternaire, avec preuves avant généralisation

Date : 25 septembre 2026. Remplace V8 comme plan actif.
Statut : **plan préparé ; méthode V9 non entraînée, résultat non acquis**.

## 0. Décision et probabilité demandée

La voie retenue est une **calibration progressive du quantificateur strict,
par fenêtres recouvrantes**, avec distillation du teacher ARC sur les états
réellement rencontrés, puis récupération globale et validation audio native.
On repart des poids denses originaux ; les records TTQ V7 restent des témoins,
pas les masters du nouveau modèle.

Événement estimé : obtenir, avec cette campagne locale bornée, un DiT complet
conforme à `W=s·q`, ≤500 000 000 octets, entraînable sous 12 000 000 000 octets
Metal, passant **tous** les critères numériques et musicaux de la section 8.

**Estimation de travail : 10–30 % de réussite de cet événement avant pilote.**
C'est un jugement prudent, subjectif et non calibré, **pas une probabilité
mesurée ni un intervalle de confiance**. Les preuves actuelles ne permettent
pas une estimation statistique fiable. Le taux demandé **>90 % n'est pas
établi**, et ce plan ne prétend pas le satisfaire aujourd'hui.

Pourquoi rester prudent : aucun modèle complet accepté ; le meilleur témoin
est partiel et non strict ; son terminal de développement est à 0,770 ; le
transfert des nouvelles méthodes LLM vers ce DiT audio n'est pas démontré ;
la calibration multi-blocs sous 12 GB reste à mesurer. Inversement, la taille
G32 tient arithmétiquement, le teacher et les masters existent, des méthodes
ternaires fonctionnent ailleurs et plusieurs erreurs de protocole sont
identifiées. Ce sont des raisons d'essayer une méthode différente, pas une
garantie.

La boucle demandée est intégrée en section 9. Répéter des raisonnements ou
des essais corrélés ne fait pas mécaniquement dépasser 90 %. Seules de
nouvelles preuves permettent de réviser l'estimation. Une réussite partielle,
une loss basse ou un pipeline sans crash ne valent pas réussite du plan.

## 1. Ce que V9 change réellement

Le [registre V9](../TERNARY_V9_EVIDENCE_2026-09-25.md) donne les preuves,
les chemins, les mesures revérifiées et les limites du diagnostic.

| Leçon V1–V8 | Conséquence V9 |
|---|---|
| Bonsai/Hadamard : bons scores locaux, effondrement global | Aucune promotion depuis un seul score de bloc ; pas de rotation dans la voie stricte |
| Fréquences/timesteps castés, biais/export/reprise mal suivis | Teacher golden, dtypes contractuels, gradient checkpointing testé, export exact dès le pilote |
| Warm-start depuis des codes et optimisation courte insuffisante | Masters denses originaux immuables ; reprise des vrais états d'optimiseur |
| TTQ améliore un bloc mais change le format final | Une scale positive, zéro exact, niveaux opposés ; pas de centre en sortie |
| Profil V8 : 256 états réels bruités, zéro trajectoire | Échantillonnage explicite par source d'état, prompt, seed, sigma et durée |
| MSE diagonale de poids en baisse, velocity en chute | Distillation de fonction ; covariance comme contrôle d'initialisation, jamais preuve de qualité |
| Tests et développement confondus | Train pour gradients ; dev pour choix ; test final scellé |
| Reprise systématique du dernier checkpoint | Meilleur checkpoint **dur rechargé**, sous contraintes, sauvegardé avec son état complet |
| Bloc 0 répété, extrapolation aux 24 blocs | Tests d'entrée/milieu/sortie et des petites matrices sensibles avant campagne complète |

Corrections de conclusions précédentes : un symlink n'est pas une fuite de
données ; des activations variables avec sigma ne rendent pas des poids
statiques impossibles ; peu de flips ne prouve pas l'absence de progrès ; un
échec d'un STE particulier ne condamne pas tous les apprentissages ternaires.

## 2. Contrat final, sans échappatoire de format

Nom du livrable : **DiT-matrices-ternary**, pas « pipeline audio entièrement
ternaire ». Le text encoder et le codec restent séparés et leur taille doit
être affichée. Les scales, biais vectoriels, normes/gates et buffers déclarés
restent continus. Si « tous les scalaires, scales comprises, dans {-1,0,+1} »
est exigé, ce serait un autre problème, non résolu par ce plan.

Pour chaque groupe de poids du périmètre :

```text
q ∈ {-1, 0, +1}
scale > 0, stockée en FP16
W_effectif = scale × q
```

Interdits dans l'artefact final : quatrième code actif, offset de groupe,
`scale_pos != scale_neg`, LoRA/résidu dense, mélange avec W_dense, recours au
teacher, poids ternaires seulement dans une base Hadamard. Les biais
vectoriels préexistants sont distincts des offsets de groupe.

Le scope est un inventaire nominatif, pas `if name.endswith('.weight')` :
les 230 tenseurs de dimension ≥2 du NPZ courant comprennent les 168 matrices
attention/FFN, le conditionnement local, les projections d'entrée/sortie et
temporelles/globales, les convolutions, les memory tokens et la projection
`cond.seconds_total_weight`. Classifier chacun avant conversion ; aucun
tenseur appris non couvert ne doit disparaître du dénominateur.

Les fréquences Fourier, rotations positionnelles, timesteps et constantes
numériques sont des buffers à préserver, pas une cible de compression.
Pour `cond.seconds_total_weight`, adapter aussi son chargeur : réutiliser le
conditionnement global dense mis en cache à la livraison cacherait un oubli.

### Budget de stockage

Comptabilité revérifiée, sans compression ZIP supposée ni headers :

| Groupes uniformes | Payload théorique | Décision |
|---|---:|---|
| G16 | 546 456 608 octets | Ne tient pas intégralement |
| G32 | 455 818 272 octets | Point de départ |
| G64 | 410 720 288 octets | Contrôle moins coûteux |
| G128 | 388 613 664 octets | Contrôle, pas choix qualité par défaut |

G16 sur certaines matrices et G32 ailleurs reste **strictement ternaire**.
Allouer au plus 40 MB supplémentaires aux groupes fins, après calcul exact
du padding et du conteneur. Ne pas confondre tailles de groupes mixtes et
poids de précision mixte. Le paquet final complet doit rester ≤500 000 000
octets, sidecars indispensables compris. Le packing 2 bits est acceptable ;
« 1,58 bit d'information » ne signifie pas 1,58 bit réellement stocké.

## 3. Recherche vérifiée au 25 septembre 2026

| Source primaire | Ce qu'on peut en utiliser | Limite pour ce projet |
|---|---|---|
| [CAT-Q, ICML 2026](https://arxiv.org/abs/2606.26650) | Modulation de l'affectation, transition lisse vers le dur, reconstruction de fenêtres ; centre utilisé pour choisir les codes, absent du poids final | LLM, pas SA3. Sur Qwen3-1.7B, moyenne 51,01 contre 61,42 en dense ; pas de quasi-équivalence universelle. Expériences annoncées sur 8 A100-80GB |
| [BitTern officiel](https://github.com/IntelChina-AI/BitTern/blob/main/projects/cat-q/README.md) et [quantificateur](https://github.com/IntelChina-AI/BitTern/blob/main/projects/cat-q/quantize/quantizer.py) | Référence d'inférence et de packing à comparer au port | Le README indique que le code d'entraînement sera publié séparément ; le quantizer consulté est inference-only. V9 sera une adaptation, pas une reproduction officielle annoncée à tort |
| [PTQ for Audio Diffusion Transformers, 2025](https://arxiv.org/abs/2510.00313) | Couvrir les pas de débruitage et comparer les distributions d'activation | Stable Audio Open, W8A8/W4A8, parfois branche low-rank ; aucune preuve de SA3 strict ternaire |
| [Stable Audio 3, mai 2026](https://arxiv.org/abs/2605.17991) | Distinguer préentraînement flow et post-entraînement adversarial | Ne pas remplacer le teacher ARC/ping-pong par BASE/Euler sans nouvelle tâche |
| [TerDiT, révision avril 2025](https://arxiv.org/abs/2405.14854) | Preuve que des DiT ternaires peuvent apprendre | Images et entraînement natif ; pas une recette de conversion audio locale |
| [Bonsai Image, mai 2026](https://prismml.com/news/bonsai-image-4b) | Exemple concret de diffusion fortement compressée | Environ 5 % de supports sensibles restent FP16 ; le 95 % de performance annoncé n'est ni strict tout-matrices ni une probabilité de conversion |
| [Bonsai 2 27B, 17 septembre 2026](https://prismml.com/news/bonsai-2-27b) | Résultat récent encourageant pour le ternaire avec scales | 98,2 % de rétention agrégée déclarée sur un LLM 27B ≠ 98,2 % de chance de réussir SA3 1,45B |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Contrôle d'initialisation utilisant l'information de second ordre | L'adaptation à une grille strictement ternaire et aux activations SA3 doit être évaluée ici |

La nouveauté utile n'est donc pas « remettre Hadamard ». C'est une affectation
plus adaptable, moins de paramètres entraînables et des fenêtres conscientes
de leurs voisins. Aucune source consultée ne valide toutes nos contraintes
simultanément. Cette sélection n'est pas une revue systématique exhaustive.

## 4. P0 — fermer les erreurs de mesure avant d'apprendre

Livrable : un `experiment_contract_v9.json`, schéma versionné, obligatoire
pour cache, entraînement, audit, export et rendu. Il lie :

- SHA-256 du teacher, des autres poids utilisés, du runtime et du code local ;
- versions Python/NumPy/MLX, configuration canonique et seed de shuffle ;
- IDs logiques, chemins réels, hashes latent/métadonnées/source/PCM quand
  disponibles, filiation parent et rôle de chaque exemple ;
- IDs exacts des prompts sélectionnés, durée demandée et crop latent ;
- source d'état : réel bruité, trajectoire teacher ou trajectoire student ;
- sigma exact FP32, indice du pas, bruit initial et toutes les réinjections ;
- hashes des caches et dtypes par tableau, scope et format de quantification ;
- historique des consultations du dev/test et parent du checkpoint.

Le validateur doit refuser un fichier supplémentaire non déclaré ou charger
exclusivement la liste manifestée. Les symlinks sont autorisés si leur cible,
hash et rôle sont vérifiés. Des noms identiques/différents ne suffisent pas à
établir l'identité des parents. Une filiation inconnue reste `unverified`.

Tests obligatoires, positifs **et négatifs** :

1. Teacher identique avec/sans instrumentation ; mêmes 16 états golden,
   répartis sur les sigmas, vérifiés dans deux processus.
2. Cache dense exact : conserver les dtypes, calculer une référence sans
   sérialisation et comparer au rejeu. Ne pas caster tout en FP16.
3. `T_lat = nombre_tokens − 64`, pas dimension des canaux. Tester 128 et une
   autre longueur pour ne pas valider accidentellement un seul shape.
4. Paramètres explicitement passés à la recomputation MLX : gradients et
   update équivalents avec/sans checkpointing sur une fixture déterministe.
5. Sauvegarde/reprise : même prochain batch, même RNG, mêmes moments Adam,
   même scheduler, mêmes poids et codes après l'update suivant.
6. Export : égalité exacte de tous les codes après unpack, scales FP16
   identiques, biais complets, aucune clé manquante/inattendue non autorisée.
7. Timestep FP16, cache d'un autre teacher, inversion de split, modification
   d'un latent, donnée ajoutée, dtype changé : chacun doit faire échouer le
   préflight avant chargement lourd.
8. Champ `heldout` dérivé du seul contrat vérifié ; aucune deuxième voie
   contradictoire. Les jeux déjà consultés sont marqués dev.

Pour les parités de représentation : codes/scales exacts ; pour les forwards,
erreur L2 relative ≤1e-3 et cosinus ≥0,99999 sur chaque état, avec maximum
absolu aussi publié. Ces seuils ne sont pas des seuils qualité. Les quatre
rejeux V8 mesurés passent la tolérance L2, mais ne remplacent pas ces tests.

**Sortie P0 :** `measurement_contract_pass`. Un échec reste un problème
d'implémentation ; ne pas lancer une nouvelle expérience de qualité dessus.

## 5. P1 — données, teacher et témoins comparables

### Rôles des ensembles

- **Train/calibration** : gradients, choix des affectations, estimation des
  sensibilités, sélection des exemples difficiles.
- **Dev** : choix de méthode, LR, groups, fenêtre et checkpoint. Les anciens
  ensembles régulièrement examinés restent dans cette catégorie.
- **Test scellé** : un seul modèle final choisi avant son ouverture. Aucune
  sélection de flips ou de checkpoint dessus ; un test qui sert ensuite à
  corriger devient du dev et doit être remplacé pour la conclusion finale.

Réconcilier les 434/33/41 existants avec la filiation réelle. Compléter le
dev vocal par d'autres styles et réserver des parents/prompts nouveaux pour
le test final. Ne pas recycler le même enregistrement via un autre crop.
Les corpus autorisés restent dans le cadre d'étude personnelle locale ; pas
de transfert externe ni de redistribution automatique.

### Couverture progressive

Pilote : 32 prompts diversifiés × 4 seeds × 8 pas = **1024 états teacher**,
plus 512 états de latents réels bruités. Après un premier candidat, ajouter
512 états issus de ses propres trajectoires, teacher réévalué dessus.
Les quotas doivent être vérifiés sur les états effectivement consommés,
pas seulement inscrits dans un manifest.

Extension si le pilote progresse : 128 prompts × 4 seeds × 8 pas = 4096
états teacher, 2048 réels bruités et 2048 student. Ce sont des budgets de
départ, pas une preuve que « 512 exemples suffisent » ni 8192 observations
indépendantes. Ajouter des données seulement si l'écart train/dev ou la
couverture manquante le justifie.

D'abord crop latent 128 ; puis fenêtres correspondant à 12/30 secondes et
contexte de production. Mesurer la longueur réellement décodée : un crop
128 et une condition « 12 secondes » ne définissent pas à eux seuls un WAV
de 12 secondes. Pour les données de tâche audio-to-audio/inpainting, couvrir
aussi le conditionnement local non nul avant de revendiquer ces usages.

Teacher figé : poids ARC, sampler ping-pong et recette de production
identiques, sans LoRA sauf si elle fait explicitement partie de la référence.
La cible principale est `teacher(x, sigma, conditions)`, pas automatiquement
`epsilon − latent_reel` d'un objectif flow de préentraînement différent.

### Témoins à conserver

1. Dense original et dense instrumenté : contrôle zéro modification.
2. W8 puis W4 **poids seulement**, activations à leur précision native :
   contrôles de faisabilité de la chaîne, jamais livrables ternaires.
3. Strict `s·q` G32 direct : vraie baseline du nouveau contrat.
4. TTQ V7 : témoin utile mais explicitement non strict et partiel.

Même sélection explicite, mêmes seeds et mêmes bruits pour chaque paire.
Le minimum dépend du nombre de cas : ne pas comparer des minima sur des
ensembles différents. Rapporter moyenne par famille, p05 et pire cas nommé.

## 6. P2 — pilote de conversion qui peut échouer vite et utilement

### 6.1 Paramétrisation principale : apprendre le quantificateur

Conserver `W0` dense figé. Apprendre par groupe trois variables continues
pour affecter les codes : déplacement `mu`, scale positive `alpha`, seuil
positif `delta`. Le déplacement agit **seulement sur la décision** :

```text
u = (W0 - mu) / alpha
q = -1 si u < -delta ; 0 si |u| <= delta ; +1 si u > delta
W_hard = fp16(alpha) * q
```

Il n'y a pas de `+mu` dans `W_hard`. Les égalités au seuil, la saturation et
le comportement d'arrondi sont spécifiés dans les fixtures du port. Le
[quantizer officiel](https://github.com/IntelChina-AI/BitTern/blob/main/projects/cat-q/quantize/quantizer.py)
sert de référence d'inférence ; fixer sa révision avant toute comparaison.

Le modèle apprend d'abord ces petits paramètres plutôt qu'Adam sur tous les
poids. Ce choix réduit la mémoire et teste si le défaut vient surtout de
l'affectation. Il ne garantit pas une capacité suffisante de réorganisation.

Transition lisse candidate, inspirée de CAT-Q :

```text
f(u, k, delta) = [tanh(k*(u-delta)) + tanh(k*(u+delta))] / [2*tanh(k)]
W_lambda = (1-lambda)*W0 + lambda*alpha*f(u, k, delta)
```

L'interpolation est une adaptation V9, pas un résultat publié pour SA3.
Elle impose l'identité dense exacte à `lambda=0`, même si `mu` est non nul.
Éviter `k=0` et les divisions instables. Augmenter `lambda` et `k` seulement
après une validation du **forward hard séparé**. À la fin : `lambda=1`,
suppression de la branche dense et remplacement de `f` par les codes exacts.

Pendant la phase dure, utiliser un surrogate de gradient explicite et testé,
sans différencier directement les entiers. Le forward est toujours
`fp16(alpha)*q` ; le choix du surrogate et sa pente sont enregistrés. La
référence publiée ne fournit pas ici de trainer prêt à porter : vérifier
gradients, bornes, hardening et export sur un problème synthétique avant SA3.

Au moins **20 % du budget d'updates en hard pur**, sans résidu. Un excellent
score soft n'autorise ni la cascade ni la livraison.

### 6.2 Objectif : même fonction, même état

Les pertes sont normalisées par l'énergie teacher avec un plancher enregistré :

```text
Nv = ||v_student - v_teacher||² / max(||v_teacher||², epsilon)
Nb = ||h_student - h_teacher||² / max(||h_teacher||², epsilon)
Nd = 1 - cosine(v_student, v_teacher)
L_initial = 0.60*Nv + 0.30*Nb + 0.10*Nd
```

Ces coefficients sont des hypothèses initiales, pas des constantes optimales.
La velocity inclut le suffixe du DiT ; son gradient doit traverser les blocs
gelés, sans les rendre entraînables. La reconstruction locale compare teacher
et student sur **la même entrée de fenêtre**, notamment les entrées issues du
préfixe étudiant. Elle ne remplace pas la cible globale `T(x)`.

Les caches d'entrée de fenêtre sont invalidés quand son préfixe change.
Les états student proviennent du checkpoint courant ; les cibles deviennent
`T(x_student)`, pas des sorties teacher prises à un autre état. Le rollout
apparié complet reste un gate ; un terme 2-pas ne s'ajoute que par ablation
après un gain pointwise dur, avec un coût mémoire mesuré.

Sampler équilibré par familles, sigma et source d'état. Garder les erreurs
RMS/DC et les faibles sigmas visibles ; un bon cosinus peut masquer un gain
erroné. Les logits d'attention différentielle et la sortie GLU servent au
diagnostic si une projection particulière reste instable.

### 6.3 Fenêtres recouvrantes, pas blocs définitivement verrouillés

Commencer par le bloc 0 puis une fenêtre `[0,1]`. Dans cette fenêtre, le bloc
0 peut être ajusté de nouveau : sa fonction ne doit pas être figée sur la
seule distribution dense. Comparer à budget égal fenêtre 1 versus fenêtre 2.
On ne généralise pas tant que **les deux blocs sont réellement hard** et que
l'audit global ne passe pas.

Ensuite, tester indépendamment les blocs 12 et 23, et les supports
temporels/conditionnement les plus sensibles. Le pilote d'un bloc complet
inclut ses projections locales, pas seulement les sept matrices historiques.
Tester tôt memory tokens et `cond.seconds_total_weight` évite de découvrir
un verrou du périmètre intégral après 24 blocs de calcul.

Fenêtre 4 seulement si fenêtre 2 montre un bénéfice plafonné et le préflight
mémoire la permet. Ce n'est pas le premier essai.

### 6.4 Ablations limitées

Maximum quatre configurations initiales, sur la même sélection :

| Essai | Question isolée |
|---|---|
| A : strict G32, calibration scales/seuils, hard dès le départ | Témoin sans transition |
| B : strict G32, transition + affectation déplacée | L'affectation et le hardening progressif gagnent-ils ? |
| C : B avec G16 sur les matrices sensibles, coût exact déclaré | Le plafond vient-il de la résolution par groupe ? |
| D : B sur deux blocs recouvrants | L'interaction des blocs explique-t-elle le plafond ? |

Garder un contrôle synthétique connu représentable en ternaire : échec à le
surapprendre = défaut d'optimisation, pas verdict sur SA3.

Un seul contrôle de remplacement est prévu si ces essais plafonnent :
initialisation par covariance amortie `H=E[xxᵀ]+eta*I`, approximation par
tuiles et propagation d'erreur de type GPTQ, **grille strictement `s·q`**.
Il ne s'agit pas de tester des millions de flips par un forward complet.
Le gain de reconstruction linéaire reste un filtre, jamais la promotion.

Si même le train représentatif ne passe pas, tester une unique récupération
avec masters entraînables **dans la fenêtre active seulement**, LR réduit,
en conservant la meilleure initialisation et le même forward hard. Ne pas
réinitialiser depuis des poids déjà arrondis.

### 6.5 Budgets du pilote

Réglages initiaux à inscrire dans la configuration, puis à mesurer :

- batch 1, accumulation 4, crop 128, gradient checkpointing explicite ;
- Adam sur paramètres de quantification, LR initial `3e-4`, clipping norm 1 ;
- ablation unique LR `1e-4` si instabilité, pas une grille ouverte ;
- première passe 500 updates par configuration, audits hard au départ puis
  toutes les 50 updates ; max 2000 updates pour le meilleur essai progressant ;
- refaire le meilleur essai sur un second seed d'optimisation ;
- deux audits hard consécutifs en régression : restauration du meilleur état,
  analyse de la cause, pas prolongation automatique ;
- enveloppe pilote : 12 h de calcul local maximum, à affiner avec le temps
  réel par update et par audit. C'est une borne proposée, pas un benchmark.

**Gate pilote** sur dev fixe, après pack/reload : velocity mean ≥0,95,
min ≥0,85 ; terminal 8-pas mean ≥0,90, min ≥0,80 ; ratios RMS [0,90;1,10]
sur les agrégats par famille ; pas de nouvel échec audio manifeste. Examiner
aussi les deltas par prompt. Les deux seeds doivent satisfaire le gate ; un
seed chanceux ne suffit pas.

Un gate manqué déclenche la section 9, pas la conversion des 24 blocs.

## 7. P3–P5 — extension, mémoire et export

### P3 : progression cumulative

Uniquement après le pilote : `[0,1] → [1,2] → ... → [22,23]`, avec le
préfixe déjà ternaire dans le vrai forward et les entrées student recalculées.
Sauver le meilleur état global à chaque fenêtre. Évaluer sur le même dev aux
jalons 2, 4, 8, 12, 18 et 24 blocs ; canary dur entre les jalons.

Un ajout de blocs doit maintenir les gates, pas seulement améliorer le nouveau
bloc isolé. Une régression rouvre la dernière fenêtre et, une fois au maximum,
la fenêtre précédente. Si le seuil reste manqué, rollback au dernier jalon
accepté et arrêt de cette branche.

Intégrer les supports inventoriés dès que leur pilote a passé ; refaire les
conditions student lorsque leurs projections changent. Après couverture
complète, une seule passe de récupération recouvrante, puis un ajustement
global des scales seulement si son coût et son gradient sont validés.
Aucun apprentissage Adam global de 1,45 milliard de masters sous 12 GB.

### P4 : politique de ressources

- Contrôle en **octets** : alerte/arrêt préventif à 11 000 000 000 Metal,
  plafond contractuel à 12 000 000 000. Publier aussi les valeurs en GiB.
- Enregistrer pic actif, mémoire allouée/réservée disponible, RSS, pression
  système, swap avant/après ; pas de double comptage CPU/Metal sur mémoire
  unifiée. Un pic sous le seuil ne prouve pas l'absence de pression système.
- Pré-calculer le text encoder, puis le décharger. Ne charger le codec que
  pour les audits audio. Teacher évalué séparément, cibles sur disque/CPU.
- Masters/Adam pour la fenêtre active uniquement. Préfixe/suffixe gelés ;
  checkpointing des activations avec test de gradient, pas `stop_gradient`
  accidentel sur le suffixe.
- Un processus par essai. Mesurer initialisation, forward, backward, update,
  audit et export ; l'export peut avoir son propre pic.
- Shards de 256 états ; conditionnements partagés, pas recopiés pour chaque
  sigma. Ne pas conserver les activations des 24 blocs simultanément.
- Pas de `q.tolist()` sur des dizaines de millions de codes : hashes binaires
  streamés. Checkpoints atomiques `best`, `resume`, `last_safe`, rotation bornée
  de dérivés uniquement ; aucune suppression automatique d'anciens résultats.
- Disque : prévoir ≥40 GB libres pour le pilote, ≥60 GB avant extension,
  avec réserve système de 10 GB. Réestimer sur un shard réel. Le contrôle
  actuel annonce 9,4 GiB : cette condition n'est **pas satisfaite**.
- Campagne complète proposée : plafond 96 h de calcul local, pilote inclus.
  Avant P3, extrapoler coût update × updates × fenêtres + audits, marge ×2.
  Si hors enveloppe, ajuster le plan explicitement, pas lancer sans borne.

L'environnement actuel permet des sondes MLX via Python 3.12.6/MLX 0.31.2,
mais l'ancien `.venv` n'existe plus. Verrouiller un environnement reproductible
et vérifier les hashes ; ne pas supposer le chemin historique opérationnel.
Pas de GPU payant, installation distante ni gros téléchargement implicite.

### P5 : paquet autonome strict

Exporter depuis le meilleur **hard** retenu. Chaque tenseur appris du scope
est présent une fois, avec shape, taille originale, padding, groupe, codes et
scale. Tester convolutions, entrées 257 non divisibles, memory tokens,
conditionnement de durée et biais. Retirer le padding avant calcul.

Recharger dans un processus propre auquel aucun checkpoint dense du DiT n'est
fourni. Le text encoder et le codec déclarés restent autorisés. Le chargeur
ne doit ni chercher un teacher pour compléter une clé, ni reconstruire le
modèle depuis un checkpoint d'entraînement. Contrôler les fichiers réellement
ouverts et toutes les clés consommées.

Un cache FP16 temporaire de déquantification n'annule pas le format des poids,
mais son coût doit être publié. « Paquet compact », « faible mémoire » et
« kernels ternaires rapides » sont trois propriétés différentes : mesurer
latence et mémoire, sans promettre une accélération par le seul packing.

## 8. P6 — critères de réussite, verrouillés avant le test final

Les seuils ci-dessous sont des **décisions de projet**, pas des constantes
issues des articles. Ils sont figés avant optimisation ; aucune baisse de
seuil pour faire passer un modèle. Comparer les quantités appariées à mêmes
prompts, bruits, durée, sampler, text encoder et codec.

| Dimension | Condition de livraison |
|---|---|
| Format et portée | Inventaire complet validé ; chaque poids du scope `s·q` ; pas de branche de secours |
| Taille | Ensemble minimal de fichiers DiT ≤500 000 000 octets ; code/texte/codec et mémoire publiés séparément |
| Reload | Codes/scales exacts ; parités de forward P0 ; aucune dépendance au teacher |
| Velocity | Moyenne ≥0,95, minimum ≥0,85 ; p05 et résultats par sigma/famille publiés |
| Trajectoire appariée 8 pas | Terminal mean ≥0,90, min ≥0,80 ; teacher-on-student et dérive RMS publiés |
| Niveau relatif | Médianes par famille de RMS student/teacher dans [0,90;1,10] ; cas hors [0,75;1,25] examinés individuellement, aucun effondrement non résolu |
| Audio technique | Float32 brut conservé, fini, durée correcte, stéréo correcte, aucun nouveau silence/statique/troncature ni écrêtage caché |
| Audio musical | Écoute A/B masquée et appariée : pas de défaut rédhibitoire nouveau ; borne inférieure 95 % de la différence de qualité moyenne >−5 points sur 100 ; conformité au prompt incluse |
| Ressources | Aucun dépassement du plafond pendant les runs mesurés ; pression système et swap déclarés |

Pour l'écoute, noter séparation/définition, transitoires, grave, aigus,
stéréo, continuité et conformité au prompt ; une note globale n'efface pas un
défaut rédhibitoire. Utiliser un bootstrap **par famille/source indépendante**,
pas par token ou pas de diffusion. Si l'intervalle est trop large, résultat
`inconclusive`, jamais réussite par défaut.

Test proposé : au moins 40 unités de tâche indépendantes, stratifiées sur
8 familles pertinentes, 2 seeds par unité. Le parent/session et la famille
de prompt définissent les clusters ; plusieurs crops ou seeds ne deviennent
pas des unités indépendantes. Sur une sous-sélection fixée d'avance, rendre
30 secondes et au moins six paires de 180 secondes pour vérifier la tenue
longue. Les autres cas testent la durée courte de référence. Les usages
audio-to-audio/inpainting ne sont annoncés que si leurs propres cas passent.

Pas d'EQ, compresseur, débruiteur, limiteur ou normalisation variable pour
réparer la sortie. Une copie à gain constant est permise pour l'A/B à niveau
égal, avec gain enregistré ; elle ne remplace ni le brut ni son audit.
L'acceptation musicale exige l'écoute humaine ; sans elle, statut
`technical_pass / musical_review_pending`, pas « qualité démontrée ».

### Ce que « >90 % » pourrait prouver après livraison

Pour **un artefact déjà figé**, 40 succès sur 40 tâches réellement
indépendantes donnent une borne binomiale unilatérale à 95 % d'environ
`0,05^(1/40) = 92,78 %`. Il faut une définition binaire de succès fixée avant
test et un échantillonnage représentatif. Les corrélations, une sélection
opportuniste ou un arrêt dès que le chiffre plaît invalident cette lecture.

Cela concerne la fiabilité du modèle sur la population testée. **Ce n'est
pas la probabilité que la recherche/formation V9 aboutisse.** Ni 40 rendus
d'un seul prompt ni 98 % de rétention d'un autre modèle ne permettent cette
substitution. Aucun tel résultat n'a été obtenu ici.

## 9. Boucle d'amélioration — changer une cause, pas annoncer un chiffre

Revue du plan déjà effectuée en trois passes :

1. **Audit historique/code** : corriger l'interprétation des symlinks, le
   statut du held-out, la couverture des trajectoires, les unités et le scope.
2. **Revue des méthodes** : écarter le solveur V8 par MSE diagonale comme
   voie principale ; retenir une adaptation progressive strictement `s·q`.
3. **Revue de faisabilité** : vérifier la taille réelle théorique, la présence
   du teacher/MLX, le petit rejeu dense et le manque de disque ; ajouter
   fenêtres recouvrantes, enveloppes de ressources et test scellé.

Ces passes améliorent la spécification ; elles n'ajoutent aucun succès de
conversion et ne remontent donc pas artificiellement la probabilité.

Boucle expérimentale à exécuter après implémentation :

```text
vérifier P0
pour au plus 3 rounds de pilote, dans le budget total :
    inscrire une hypothèse falsifiable et son témoin
    exécuter le plus petit test qui les départage
    packer/recharger puis mesurer le hard sur dev fixe
    si bug / mauvaise provenance : invalider le run, corriger puis rejouer
    si train gagne mais dev perd : traiter couverture/surapprentissage
    si soft gagne mais hard perd : traiter transition/quantificateur
    si bloc gagne mais DiT perd : traiter suffixe/fenêtre recouvrante
    si velocity gagne mais audio perd : traiter trajectoire/codec/objectif
    si taille/mémoire dépasse : corriger comptabilité ou scheduling
    conserver la meilleure preuve, journaliser aussi les échecs
    réévaluer la plausibilité sans supposer les essais indépendants
    si tous les gates pilote passent sur 2 seeds : autoriser P3
    sinon ne pas ouvrir de cascade complète
```

Round 1 : calibration progressive A–D. Round 2 seulement si justifié :
couverture on-policy ou covariance d'initialisation, selon la cause observée.
Round 3 : récupération limitée des masters de fenêtre, seulement si les
précédents tests isolent un plafond de capacité du quantificateur.

Si le pilote strict reste rejeté après ces rounds ou si le budget est atteint :
statut **`strict_target_not_demonstrated`**, résultat conservé, aucune release.
Cela ne prouve pas une impossibilité mathématique. Augmenter le compute,
changer de base, entraîner nativement, conserver des matrices W4 ou ajouter
un résidu sont des variantes à discuter séparément, pas des succès déguisés
du plan actuel. Une boucle sans hypothèse nouvelle ne doit pas consommer
indéfiniment du calcul pour satisfaire un nombre demandé.

## 10. Livrables d'implémentation et première action

Les noms nouveaux ci-dessous sont **à implémenter**, pas des commandes déjà
disponibles. Les outils V8 ne doivent pas être lancés comme s'ils satisfaisaient
déjà ce contrat.

| Lot | Fichiers/capacités à produire | Vérification de sortie |
|---|---|---|
| P0 | `ternary_experiment_contract_v9.py`, unification audit/cache, tests de provenance/parité/reprise | Tests négatifs refusés, golden dense stable |
| P1 | Sélections nominatives, cache stratifié et manifest des bruits/dtypes | Quotas réellement consommés, aucune fuite prouvée ou non résolue |
| P2 | `train_ternary_calibration_v9.py`, quantizer strict, hardening, contrôles A–D | Pilote hard reload accepté sur deux seeds |
| P3 | Scheduler de fenêtres recouvrantes, refresh student, Pareto checkpoints | Jalons cumulatifs acceptés |
| P4/P5 | Resource guard, export autonome et inventaire toutes matrices | Octets, couverture et reload sans teacher conformes |
| P6 | Rapport test, WAV bruts, fiche A/B, décision humaine | Tous critères section 8 ; sinon non livré |

Chaque run doit contenir config + digests, `train.jsonl`, `dev.json`,
`hardening.json`, mesures ressources, checkpoint complet de reprise, records
de déploiement, journal des décisions et motif d'arrêt. Les décisions passent
dans la [base de connaissances](../TERNARY_DISTILLATION_KNOWLEDGE_BASE.md).

**Prochaine action utile : P0 et le banc synthétique strict, après réserve
disque suffisante. Pas un entraînement 24 blocs immédiat.** Le présent travail
a produit un plan, un audit et des sondes de lecture ; il n'a pas lancé cette
campagne. La probabilité >90 % demandée reste explicitement non démontrée.
