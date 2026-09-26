# Plan V7 — ternarisation qualité du DiT Stable Audio 3

> Document historique. Le plan actif est désormais [V9](TERNARY_QUALITY_RECOVERY_PLAN_V9.md).

Date de révision : 24 septembre 2026  
Statut : **plan historique, aucun modèle final accepté**.  
Périmètre : DiT Medium MLX, poids ternaires `{-1, 0, +1}`, Mac Apple Silicon,
garde Metal 11 Go.

Ce document remplace l’idée « prolonger P3.1 puis ouvrir la cascade ». P3.1
a bien exécuté le pipeline, mais il n’a pas produit la fidélité requise. La
prochaine tentative doit changer la paramétrisation et le protocole de
sélection, pas seulement augmenter `steps`.

## 1. Verdict factuel des essais précédents

Les audits valables sont ceux exécutés avec le corpus du cache :
`output/sample-expertise-pilot/ternary-quality-v6-20260923/authorized-independent-sftvoices-v1/train`.
Les audits des nouveaux artifacts faits avec `universal-dataset/latents-12s`
étaient sur un autre corpus et sont conservés comme trace, mais exclus de toute
comparaison.

| Run | Procédure | Velocity mean/min | Terminal mean | Audio brut cache | Pic Metal |
|---|---|---:|---:|---|---:|
| P2 v3 | dense → symmetric G32, 50 updates | 0,93262 / 0,78394 | 0,82143 | rouge, RMS 2,002 / 1,593 / 1,796 | 6,01 Go |
| Contrôle A | warm-start FP32/Adam P2, 112 updates, pointwise | 0,92804 / 0,73204 | 0,75743 | rouge, RMS 1,483 / 1,558 / 1,342 | 6,44 Go |
| P3.1 | même warm-start, `Lv + 0,25 endpoint`, 224/224 paires | 0,92819 / 0,73154 | 0,75873 | rouge, RMS 1,463 / 1,534 / 1,326 | 6,92 Go |
| learned-symmetric pilote | projection records puis 16 updates | 0,91795 / 0,71337 | 0,70638 | non promu | 8,94 Go |

Les trois ratios audio de référence sont respectivement `dub`, `voice`,
`classical_piano`; aucune normalisation n’a été appliquée. Les artefacts A et
P3.1 ont passé le round-trip cross-process et couvrent le cache de paires. Le
problème n’est donc pas un export corrompu ni une reprise Adam incomplète.

Résultat complémentaire : P3.1 ne gagne que `+0,015 %` de cosine moyen sur A,
avec un minimum légèrement pire. La correction « conserver les maîtres FP32 »
était nécessaire, mais **pas suffisante**. Prolonger la même optimisation est
contre-productif : A/P3.1 dérivent après le P2, sans sélection de meilleur
checkpoint intermédiaire.

## 2. Ce qui a réellement échoué

### 2.1 Quantizer symmetric trop rigide

Le mode actuel choisit par groupe une échelle symétrique et une affectation dure
par `round(mean_abs)`. Le maître FP32 reçoit un STE quasi-identité, mais le
forward ne dispose ni d’un seuil appris indépendant ni d’échelles positives et
négatives séparées. Après 112 updates, seulement `68 241 / 113 246 208`
codes ont changé, soit `0,0603 %`; P3.1 est à `0,0617 %`. Les scales peuvent
bouger sans réparer les décisions de code qui déterminent la forme du bloc.

### 2.2 L’erreur de bloc se compose dans le rollout

Le modèle ne fait pas qu’une prédiction isolée : la sortie d’un pas devient
l’entrée du suivant. Une bonne loss ponctuelle moyenne masque les prompts et
les sigmas faibles où l’erreur d’état s’amplifie. Le P2 avait déjà un terminal
à `0,82143`; P3.1 optimise deux pas sur des ancres dérivées d’un candidat P2,
mais ne contraint pas suffisamment les 8 états de la trajectoire complète.

### 2.3 Pas de sélection de checkpoint

Les runs ont exporté le dernier état. Le dernier état n’est pas forcément le
meilleur : A/P3.1 finissent avec une loss remontée et des métriques inférieures
au P2 source. Toute V7 doit auditer périodiquement, conserver
`best_validation`, et arrêter/revenir dès qu’un update dégrade le minimum ou le
terminal.

### 2.4 La preuve de données doit être obligatoire

Le cache V7 porte 16 prompts et 2 seeds, mais l’audit accidentel sur un autre
dataset a quand même pu démarrer. Un artifact, son cache, son manifest de split
et l’audit doivent porter le même `dataset_sha256`/`prompt_set_digest`; sinon la
commande doit échouer avant de charger le modèle.

### 2.5 Ce que les échecs ne prouvent pas

Ils ne prouvent pas que la ternarisation est impossible. Ils prouvent que la
projection symmetric directe, suivie d’un QAT court sur deux blocs et d’une
loss de trajectoire partielle, n’est pas une méthode de qualité pour ce DiT
audio.

## 3. Recherche récente utilisée pour la V7

- [TerDiT, arXiv 2405.14854, version révisée 2025](https://arxiv.org/abs/2405.14854)
  et [code officiel](https://github.com/Lucky-Lance/TerDiT) : démontre la
  faisabilité d’un DiT ternaire avec QAT, mais par entraînement natif/from
  scratch et avec des choix d’architecture dédiés. Ce n’est pas une preuve
  qu’une simple PTQ/QAT locale préservera Stable Audio 3.
- [TTQ, arXiv 1612.01064](https://arxiv.org/abs/1612.01064) : apprend le
  seuil et les facteurs de niveaux ternaires au lieu de figer une seule
  projection. V7 doit au minimum tester des facteurs positifs/négatifs et un
  seuil par groupe.
- [QuEST, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_QuEST_Low-bit_Diffusion_Model_Quantization_via_Efficient_Selective_Finetuning_ICCV2025_paper.html) : relie la difficulté low-bit aux activations déséquilibrées et recommande une
  adaptation sélective avec supervision locale et globale. V7 ajoute donc une
  reconstruction de sortie de bloc et un profil de sensibilité par sigma.
- [TQ-DiT, arXiv 2502.04056](https://arxiv.org/abs/2502.04056) : montre que les
  distributions d’un DiT varient avec le temps et propose une calibration
  time-aware/multi-region. V7 couvre les huit sigmas de production; aucune
  calibration à un seul timestep ne peut être libérée.
- [Scheduling Weight Transitions for QAT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Scheduling_Weight_Transitions_for_Quantization-Aware_Training_ICCV2025_paper.html) : motive une transition progressive vers le forward dur au lieu d’imposer la discontinuité dès le premier update.
- [SpinQuant](https://arxiv.org/abs/2405.16406), [QuaRot](https://arxiv.org/abs/2404.00456)
  et [ConvRot](https://arxiv.org/abs/2512.03673) : les rotations de type
  Hadamard peuvent réduire les outliers avant quantification, ConvRot ciblant
  explicitement les DiT. Elles ne sont pas importées comme magie : la rotation,
  son inverse et son coût doivent être implémentés et vérifiés dans le forward
  réel avant d’être comparées.
- [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) : sépare clairement
  masters d’entraînement et poids packés d’inférence. V7 conserve cette
  séparation et ne reprend jamais un entraînement à partir de seuls codes
  arrondis.

## 4. Principe de la nouvelle méthode

La V7 suit quatre règles non négociables :

1. **Calibrer le forward du bloc, pas seulement `||W - Q(W)||`.** Le choix du
   seuil et de l’échelle est jugé sur les sorties réellement activées par
   Stable Audio 3, sur les huit sigmas et plusieurs prompts.
2. **Apprendre un quantizer expressif.** Tester un TTQ group-wise avec
   `q ∈ {-1,0,+1}`, seuil appris, `scale_pos` et `scale_neg` appris. Le mode
   symmetric reste le contrôle, pas le candidat par défaut.
3. **Passer progressivement au dur.** Démarrer avec un surrogate lisse contrôlé,
   augmenter `hardness` par paliers, puis verrouiller le forward exactement
   ternaire avant toute mesure de release.
4. **Choisir le meilleur état validé.** Un run n’exporte que le checkpoint dont
   la validation complète est la meilleure sous les gates; `window_latest` seul
   n’est jamais une preuve de qualité.

Le modèle final reste 100 % ternaire sur le périmètre DiT déclaré. Une couche
conservée en FP16 peut exister dans une ablation de sensibilité, mais elle ne
peut pas être comptée comme modèle final « ternaire ».

## 5. Plan d’exécution par gates

### P0 — verrouiller le contrat d’expérience

Livrables :

- `dataset_contract.json` liant train/validation/test, les 16 prompts du cache,
  les chemins et hashes des latents/métadonnées;
- une sélection validation indépendante, au moins 8 prompts et 2 seeds par
  prompt, jamais utilisée pour les gradients;
- audit et rendu refusant un corpus dont le digest ne correspond pas à
  l’artifact;
- cache de cibles reconstruit **après** le dernier changement du trainer, puis
  `preflight` de provenance réussi.

Gate P0 : aucun NaN, teacher déterministe, huit sigmas identiques dans cache,
train, audit et renderer, et round-trip dense/packed passé. Échec = zéro
entraînement.

### P1 — trouver une projection ternaire qui préserve un bloc

Construire un banc de calibration hors backprop lourd, toujours sur les mêmes
activations teacher :

1. symmetric G32 et G64, contrôle historique;
2. affine-centered G32/G64;
3. TTQ group-wise : seuil `τ_g`, `s+_g`, `s-_g`, affectation dure;
4. learned-symmetric actuel, seulement comme ablation;
5. rotation Hadamard par groupe puis 2–4, après intégration exacte du runtime.

Pour chaque candidat : chercher les paramètres sur les 7 projections des blocs
0 et 1, puis mesurer séparément `block_output_nmse`, velocity cosine par sigma,
minimum par prompt et terminal sur 8 pas. Le poids-MSE seul ne sélectionne rien.

Gate P1 de recherche : le candidat doit battre symmetric sur le **minimum** de
cosine et sur le terminal, pas seulement sur la moyenne. Tout candidat sans
gain clair après calibration est supprimé avant QAT.

### P2 — implémenter le TTQ/LSQ dur et son export exact

Modifier le contrat partagé, pas seulement le trainer :

- `TernaryQATLinear` : masters FP32, `τ_g`, `s+_g`, `s-_g`, forward dur et
  surrogate STE distincts;
- gradient du seuil/scale testable sur un petit layer synthétique;
- record versionné contenant codes, seuils et niveaux réellement utilisés;
- pack 2-bit avec métadonnées FP16 par groupe;
- reload séparé qui reproduit exactement le forward du layer;
- métriques de flips, fraction zéro, niveaux positifs/négatifs et distance au
  seuil.

Ne pas réutiliser le warm-start symmetric pour un quantizer dont les paramètres
ont changé. Le seul warm-start autorisé est un transfert explicite des masters
FP32 si le contrat de formes et de provenance est compatible; les états Adam
incompatibles sont réinitialisés et cette décision est journalisée.

### P3 — QAT progressif depuis le professeur dense

Un pilote sur **un bloc**, puis deux blocs seulement si le premier passe :

- 16 updates soft (`hardness` bas), 32 updates transition, 64 updates hard;
- masters et quantizer parameters séparés, AdamW FP32;
- grille courte de LR : masters `3e-6`, `1e-5`, `3e-5`; scales/seuils
  `1e-4`, `3e-4`; une seule seed pour le tri;
- mélange objectif initial :
  `0,45 L_velocity + 0,30 L_block_output + 0,20 L_one_step + 0,05 L_norm`;
- après passage hard, ajouter `L_rollout_4steps` progressivement. La loss
  trajectoire ne doit jamais remplacer la couverture velocity des huit sigmas;
- ancres générées par le teacher et états réels; ne pas recycler uniquement les
  ancres du candidat P2;
- validation toutes les 8 updates, artefact records-only léger, sélection de
  `best_validation` et arrêt si deux évaluations consécutives régressent.

Gate de promotion d’un bloc :

- velocity mean ≥ `0,95` et minimum ≥ `0,85` sur le train d’audit;
- validation mean ≥ `0,94`, minimum ≥ `0,82`;
- terminal state mean ≥ `0,90`, terminal minimum ≥ `0,80`;
- ratios RMS bruts dans `[0,70; 1,30]` et peaks dans `[0,50; 1,50]` sur 3
  prompts de développement et 3 validation;
- au moins `0,2 %` de codes effectivement modifiés pendant la transition,
  sans explosion > `5 %` en un checkpoint;
- pic Metal ≤ `11 Go`, codes/échelles/thresholds exacts après reload.

Ces seuils sont des gates de travail, pas des résultats supposés. Si le
quantizer TTQ n’atteint pas P3 sur un bloc, la cascade est fermée et il faut
revenir à P1 (group size, rotation, supervision), pas ouvrir davantage de
blocs.

### P4 — reconstruction de bloc et stabilisation du rollout

Quand P3 passe :

- distillation locale à chaque projection avec activations teacher;
- distillation globale de la velocity en sortie du DiT;
- rollout student-on-policy de 4 puis 8 pas, bruit et réinjections conformes
  au contrat;
- pondération par sigma, avec surpondération des régions qui dégradent le
  terminal, enregistrée dans le manifest;
- calibration des gains de sortie de bloc uniquement si le gain fait partie du
  runtime final et est sérialisable.

P4 doit comparer `best_validation` contre le checkpoint initial du bloc. Une
  loss plus basse sans terminal ou audio meilleur est un échec.

### P5 — cascade séquentielle contrôlée

Ternariser un bloc à la fois :

1. charger les blocs déjà acceptés en hard packed;
2. ouvrir uniquement le bloc suivant avec son master dense/TTQ;
3. distiller sur les sorties du teacher et du student cascade;
4. valider, exporter et figer le meilleur checkpoint;
5. seulement ensuite passer au bloc suivant.

La fenêtre 0–1 n’est pas un raccourci vers 0–3. Chaque extension doit conserver
les records précédents à l’identique. Une régression d’un bloc déjà accepté
rouvre son entraînement; elle n’est pas masquée par une moyenne globale.

### P6 — artifact final et preuve de déploiement

Le release candidate doit :

- contenir tous les poids DiT éligibles au format ternaire déclaré, pas seulement
  14 matrices de deux blocs;
- tenir dans la cible `≤500 000 000` octets après packing réel;
- être rechargé par un processus indépendant avec codes, seuils, scales et
  forward identiques dans les tolérances;
- passer les 8 sigmas, le rollout 8 pas, 3 seeds, train/validation séparés et
  au moins 6 rendus audio bruts;
- rester sous 11 Go Metal durant load et rendu;
- avoir un rapport de taille, mémoire, provenance, tests et écoute aveugle.

## 6. Instrumentation obligatoire à ajouter

Le trainer V7 doit refuser ou enregistrer explicitement :

- `dataset_digest`, `prompt_set_digest`, `state_cache_digest`,
  `teacher_digest`, `quantizer_contract_digest`;
- `best_validation_step`, score complet et raison d’arrêt;
- métriques par sigma, par prompt, par module et par transition;
- code flips persistants, distance au seuil, zéro fraction, `s+`, `s-`;
- séparation `train_loss`, `validation_loss`, `block_loss`, `rollout_loss`;
- mémoire active/pic avant, pendant et après export;
- hashes du record checkpoint et de l’artifact packed.

Tests minimaux avant nouveau GPU run :

- gradient finite-difference du seuil et des deux scales;
- forward dur = export = reload sur layer et bloc;
- aucune donnée validation dans les gradients;
- reprise exacte et warm-start incompatible refusés;
- mauvais dataset manifest refusé;
- sélection `best_validation` reproductible;
- `git diff --check`, `py_compile`, tests V7 ciblés.

## 7. Commande de travail et règles d’arrêt

Chaque run doit avoir un répertoire neuf et un manifest de configuration. Ne
jamais lancer deux candidats Metal simultanément. Ne jamais supprimer les
checkpoints utiles avant le rapport comparatif.

Ordre impératif :

```text
P0 provenance
  → P1 calibration quantizer
  → P2 forward/export TTQ
  → P3 pilote un bloc + validation/audio
  → P4 rollout 4/8 pas
  → P5 cascade
  → P6 release
```

Arrêt immédiat si : mauvais corpus, cache périmé, teacher non déterministe,
NaN, dépassement Metal, round-trip non exact, minimum validation en baisse,
codes immobiles, audio hors amplitude, ou artifact qui dépasse la cible. Une
gate rouge produit un rapport et une nouvelle hypothèse; elle ne déclenche pas
automatiquement plus de calcul.

## 8. Critère de réussite honnête

La V7 est réussie seulement quand un artifact complet, réellement ternaire,
est meilleur que le baseline P2 sur le minimum et le terminal **sur validation
indépendante**, puis passe l’audio brut sans normalisation. Un artifact qui
charge, compresse ou passe un audio pilote mais reste à ~`0,73` de cosine
minimum est un résultat de recherche, pas un modèle de qualité.

Artefacts V7 examinés :

- [Contrôle A](../output/sample-expertise-pilot/ternary-quality-v7-20260924/control-a-p3-1-warmstart-paired-epoch-112updates-seed20260924-fix1/window_summary.json)
- [P3.1](../output/sample-expertise-pilot/ternary-quality-v7-20260924/p3-1-trajectory-warmstart-paired-epoch-112updates-seed20260924-fix1/window_summary.json)
- [Audit A corpus cache](../output/sample-expertise-pilot/ternary-quality-v7-20260924/control-a-p3-1-warmstart-paired-epoch-112updates-seed20260924-fix1/audit-instrumental-cache-train-16/audit_summary.json)
- [Audit P3.1 corpus cache](../output/sample-expertise-pilot/ternary-quality-v7-20260924/p3-1-trajectory-warmstart-paired-epoch-112updates-seed20260924-fix1/audit-instrumental-cache-train-16/audit_summary.json)
- [Audio A corpus cache](../output/sample-expertise-pilot/ternary-quality-v7-20260924/control-a-p3-1-warmstart-paired-epoch-112updates-seed20260924-fix1/audio-pilot-3prompt-cache-train-8steps/audio_metrics.json)
- [Audio P3.1 corpus cache](../output/sample-expertise-pilot/ternary-quality-v7-20260924/p3-1-trajectory-warmstart-paired-epoch-112updates-seed20260924-fix1/audio-pilot-3prompt-cache-train-8steps/audio_metrics.json)
