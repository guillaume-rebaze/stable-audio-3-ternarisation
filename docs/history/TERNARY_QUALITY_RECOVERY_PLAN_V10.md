# Plan V10 — ternarisation Bonsai qualité

Date : 25 septembre 2026  
Statut : **package structurel réussi ; accepté localement après écoute**.

Ce document remplace V9 comme plan opératoire. Il sépare enfin deux faits :

- le format Bonsai est correctement produit et rechargé ;
- l'écoute utilisateur accepte le rendu pour l'étude personnelle ;
- la validation automatique générale reste plus stricte et échoue.

## Contrat fixe

- Modèle : Stable Audio 3 Medium, crop `128`.
- Cœur uniquement : 24 blocs × 7 matrices = **168**.
- Chaque groupe : `q ∈ {-1,0,+1}`, `W = s*q`, `s` FP16 positif.
- G32 : 2 bits par code, une scale par groupe.
- Supports natifs : **357** tenseurs, biais vectoriels inclus.
- Aucun LoRA, résidu dense, moyenne affine ou matrice de contournement.
- Hadamard autorisé seulement comme base d'entrée explicitement déclarée.
- Envelope Bonsai : **650 000 000 octets** ; cible utile 550–650 Mo.
- T5/SAME-L restent hors du package DiT, comme dans Bonsai Image.

## État réellement obtenu

Package : [`final-ternary-bonsai-g32-v10`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/final-ternary-bonsai-g32-v10/)

| Contrôle | Résultat |
|---|---:|
| Payload | 513 342 833 octets |
| Payload logique calculé | 613 897 248 octets |
| Matrices ternaires | 168/168 |
| Supports natifs | 357 |
| Codes ternaires | 1 358 954 496 |
| Paramètres après rechargement | exact |
| Forward package/reference | L2 `0.0`, cosine `1.0` |
| Envelope 650 Mo | PASS |

Preuves : `manifest.json`, `reload_report.json`,
`services/musicgen/export_ternary_bonsai_package.py` et
`services/musicgen/verify_ternary_bonsai_package.py`.

La cascade complète a aussi fini structurellement : blocs 8–23 promus après
reprise du bloc 23. Mais la promotion était lancée avec
`--promote-on-quality-fail` pour mesurer la dérive, pas pour déclarer un
modèle utilisable.

Résultat qualité final :

- velocity mean cosine : **0,876934** ; min : **0,646171** ;
- state trajectory mean : `0,955889` ; terminal mean : `0,844483` ;
- canary audio cosine : **0,852056** ; latent L2 relatif : `0,396341` ;
- `technical_pass=true`, mais `candidate_velocity_pass=false` et
  `release_velocity_pass=false`.

Les gates du code sont candidate `mean≥0,90` et `min≥0,80`, release
`mean≥0,93` et `min≥0,85`. Le résultat actuel échoue même le candidat.

## Diagnostic de l’échec V10

1. Chaque bloc est quantifié puis gelé. L’optimisation locale voit un préfixe
   déjà imparfait, mais ne corrige pas l’erreur terminale du DiT.
2. Les blocs 0–5 sont directs et les blocs suivants Hadamard. Le format est
   légal, mais cette transition n’a pas été prouvée optimale pour SA3.
3. Le surrogate `identity` permet un round-trip exact, pas une garantie de
   flips utiles. Une loss locale basse ne garantit pas le velocity field final.
4. `--promote-on-quality-fail` a volontairement continué la cascade malgré les
   gates ; la chute progressive (`~0,96` au début, `0,877` à 24 blocs) confirme
   une accumulation, pas un problème de packing.
5. Le fallback rollout complet était trop lent sur MPS : il n’a pas produit de
   checkpoint exploitable. Le plafond observé était ~4,34 Go pendant le QAT
   pointwise, donc **VRAM non bloquante**.
6. Les checkpoints cumulatifs dupliquaient tout le cœur. Ils ont rempli le
   disque et interrompu le bloc 23 ; ce défaut est maintenant documenté et le
   package compact est écrit séparément.

## Plan qui doit passer avant toute nouvelle cascade

### P0 — verrouillage structurel, déjà passé

- cache et cibles versionnés, états audit alignés ;
- codes, scales, biais et modes sauvegardés ;
- reprise inter-processus ;
- fixture hard-forward ;
- package compact indépendant ;
- reload frais + forward frais.

Ne pas refaire P0 sauf modification du runtime.

### P1 — trouver une recette qualité sur un seul bloc

Ne jamais lancer 24 blocs avec un bloc qui échoue.

Pour les blocs `0`, `6`, `12`, `18` et `23`, lancer trois candidats courts,
identiques hors seed :

1. direct symmetric, maître dense teacher ;
2. Hadamard symmetric, maître dense teacher ;
3. QAT hard avec surrogate smooth puis gel dur.

Chaque candidat est évalué sur les 512 états train, 128 états held-out, les
16 prompts et les 4 sigmas. Mesurer séparément : loss locale, flips de codes,
distance aux seuils, velocity, état terminal, audio brut. Aucun EQ, limiteur
ou normalisation réparatrice.

Pass P1 : candidate gate `mean≥0,90`, `min≥0,80`, aucun état held-out sous
`0,75`, audio technique valide. Sinon le candidat est rejeté.

### P2 — objectif terminal sans explosion VRAM

Le pointwise est conservé comme warm-start, puis un second passage optimise le
bloc actif avec une cible terminale :

- préfixe et suffixe gelés ;
- activations du préfixe cachées en cache ;
- forward suffixe recomputé par microbatch ;
- perte velocity teacher/student + état terminal + petite perte locale ;
- 32 états par microbatch, accumulation 2, AMP/FP32 identique au runtime ;
- checkpoint uniquement du bloc actif, pas du modèle cumulatif.

Si le rollout complet dépasse 11 Go ou devient trop lent, réduire le
microbatch et augmenter l’accumulation. Ne pas supprimer l’objectif terminal.

Pass P2 : amélioration mesurée sur held-out et audio, sinon retour au dernier
checkpoint accepté.

### P3 — apprentissage conjoint des blocs sensibles

Après deux blocs acceptés, rouvrir une fenêtre glissante de 2–4 blocs avec
maîtres FP32 persistants, tout en réexportant uniquement `q/s` ternaires. Le
modèle livré reste strictement Bonsai ; la jointure ne sert qu’à réallouer
l’erreur entre blocs.

Ordre : bloc le plus sensible d’abord, puis ordre DiT. Pour chaque promotion :

1. produire trois seeds ;
2. audit held-out ;
3. canary audio A/B brut ;
4. accepter seulement si le cumul ne perd pas plus de `0,005` mean cosine,
   ne tombe pas sous `0,85` min, et garde un audio valide ;
5. sinon rollback immédiat.

### P4 — choix de base et cascade 24

Comparer deux campagnes complètes :

- **A** : direct symmetric partout ;
- **B** : Hadamard partout, même group size et même recette.

La campagne mixte V10 n’est pas la référence finale. Garder la meilleure
campagne selon l’audit held-out et l’audio, pas selon la taille du fichier.

Interdire `--promote-on-quality-fail` dans la campagne release. Un seul bloc
échoué arrête la cascade et déclenche P3.

### P5 — export scellé

Exporter seulement depuis le dernier checkpoint accepté :

- supports natifs exacts, biais appris inclus ;
- core sans dense duplicate ;
- q limité à `{-1,0,+1}` ;
- code 2 bits `3` rejeté ;
- scale positive FP16 ;
- manifest avec hash teacher, code, dataset, cache et cascade ;
- payload ≤650 Mo.

Refaire dans un processus neuf : chargement, inventaire, paramètre exact,
forward exact, canary audio et test de taille. Cette procédure est passée;
le package est accepté localement pour l'étude personnelle, sans être présenté
comme une validation aveugle universelle.

## Probabilité honnête

À l’état V10 :

- réussite structurelle Bonsai : **démontrée** par le package et le reload ;
- acceptation locale d'écoute : **validée par Guillaume** ;
- réussite qualité release générale : **non démontrée**, car le gate est raté ;
- probabilité d’une prochaine recette qualité : **non quantifiable avant P1**.

Promettre `>90 %` avant un bloc P1 passé serait inventer une certitude. La
boucle correcte est donc : P1 court → mesure → P2/P3 → cascade seulement après
passage des gates. Si aucun des trois candidats P1 ne passe, arrêter et
changer de quantizer/base ; ne pas consommer le disque avec une V11 identique.

## Commandes de référence

```bash
rtk proxy python3 services/musicgen/export_ternary_bonsai_package.py \
  --records-checkpoint <accepted>/records_checkpoint.npz \
  --teacher-weights /Users/guillaumegaillard/.cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz \
  --output-dir output/.../final-ternary-bonsai-g32-v10 \
  --group-size 32 --crop-len 128

rtk proxy python3 services/musicgen/verify_ternary_bonsai_package.py \
  --package-dir output/.../final-ternary-bonsai-g32-v10 \
  --teacher-weights /Users/guillaumegaillard/.cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz \
  --dataset-dir output/.../train
```

## Verdict

Le but Bonsai de format est atteint et le livrable est finalisé pour l'étude
personnelle. Une optimisation P1/P2 reste optionnelle si une validation
générale est nécessaire; elle ne doit pas remplacer ce package accepté ni
déclencher une nouvelle cascade aveugle.
