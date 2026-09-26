# Plan v5 — student ternaire natif, joint et vérifiable

> Historique, remplacé le 23 septembre 2026 par le [plan v6](TERNARY_QUALITY_RECOVERY_PLAN.md).
> Ne pas exécuter cette proposition : indépendance du split surestimée,
> diagnostic de capacité non démontré, entraînement joint non budgété.
> Le texte ci-dessous reste conservé pour traçabilité, pas comme consigne actuelle.

Date : 22 septembre 2026. Prérequis : v4 exécuté, rejeté sur validation/test.

## Décision

Ne pas relancer le pipeline v3/v4 bloc par bloc avec davantage de données. Le
run v4 dispose maintenant d’un split indépendant et montre que l’ajout de
corpus ne corrige pas la distorsion du core strict `W_hat = s*q`.

Objectif v5 : construire un student dont la capacité est optimisée directement
pour la sortie du DiT, avec codes durs `q ∈ {-1,0,+1}` et scales positives
apprises. Le student ne doit pas être présenté comme « 100 % ternaire » tant
que chaque paramètre appris du chemin d’inférence n’est pas inventorié.

## Invariants

- Split scellé v4 : train `g2-authorized-expanded-v3/train`, validation
  `.../validation`, test `.../test` ; le test ne sert jamais au choix.
- Sources originales immuables ; autorisation locale personnelle consignée ;
  aucune redistribution.
- Teacher FP16 figé, même conditioner, codec, sampler, crop, sigmas et seeds.
- Codes exportés uniquement `{-1,0,+1}` ; aucun fallback affine silencieux.
- Toutes les matrices/Convs réellement exécutées sont listées dans le scope.
  Normes, biais et embeddings hors scope sont comptés séparément, jamais cachés.
- Chaque checkpoint contient poids maîtres/logits, scales, optimiseur, RNG,
  ordre des données, scope et hashes. Export immuable distinct du train.

## Étape 0 — inventaire de capacité

1. Énumérer `DiT` complet : Linear, Conv1d, embeddings, projections,
   modulations et biais.
2. Mesurer le budget exact pour G32/G64/G128 et le coût des scales ; publier
   tailles compressées/décompressées avant entraînement.
3. Écrire un test qui échoue si un paramètre appris du forward n’est ni dans le
   scope ternaire ni explicitement déclaré FP16 de contrôle.

Sortie : `native_scope.json`, `native_size_budget.json`, aucune QAT si le
budget dépasse 500 000 000 octets ou si le scope n’est pas total.

## Étape 1 — opérateur natif

Implémenter `TernaryNativeLinear` et, si nécessaire, `TernaryNativeConv1d` :

- logits/scale maîtres FP32 pendant le train ;
- projection STE vers trois codes durs à chaque forward ;
- scale positive par groupe, calibration séparée des outliers ;
- packing déterministe `uint32` à l’export ;
- reload indépendant puis comparaison du forward avant toute mesure qualité.

Tests obligatoires : code réservé, shape, signe, packing exact, gradient STE,
parité dense/quantifiée à zéro-step, reload dans un processus neuf et scope
complet.

## Étape 2 — pilote joint

Entraîner simultanément tous les paramètres ternaires déclarés, avec le teacher
sur le même `x`, plutôt que de figer un bloc après l’autre.

Budget pilote : blocs 0/12/23, 10 puis 50 updates, deux seeds, validation
12 prompts. Loss composée : velocity teacher/student, état latent après 1–2
pas, pénalité de saturation des scales et régularisation de l’occupation des
codes. Les poids d’un préfixe ne sont jamais remplacés par une relaxation douce
non exportable.

Gate de continuation : amélioration de validation à deux checkpoints et
cosine moyen ≥ `0,90`, minimum ≥ `0,80`. Sinon, arrêter la branche native et
conserver la preuve négative.

## Étape 3 — cascade complète conditionnelle

Seulement si l’étape 2 passe : `100` updates, puis `200` maximum, checkpoint
toutes les `25/50` étapes, validation après chaque checkpoint. Deux seeds au
maximum. La meilleure validation est exportée, rechargée et auditée avant tout
usage du test.

Gates candidat/release :

- validation mean/min ≥ `0,90 / 0,80`, puis release ≥ `0,93 / 0,85` ;
- test 24 prompts × 5 sigmas, aucune optimisation dessus ;
- audio brut 3 prompts puis 12 prompts, 12/30 s, sans mastering masquant ;
- fichier ≤ `500 000 000` octets ;
- RSS, Metal et swap mesurés sur la machine cible, sous réserve saine ;
- écoute humaine niveau égal, défauts et incertitude écrits.

## Contrôles non-promouvables

1. **Affine-centered** : diagnostic de capacité, pas strict symétrique.
2. **G32** : ablation taille/qualité ; dépassement de taille explicitement
   rejeté si confirmé.
3. **Résidu FP16** : upper bound de capacité ; modèle hybride, jamais annoncé
   100 % ternaire.

Un contrôle peut expliquer l’échec, mais ne peut pas remplacer la branche
native ni contaminer le test v4.

## Arrêt et sortie

Arrêt immédiat sur scope incomplet, NaN, code invalide, swap persistant,
overlap de parent ou absence de progrès. Aucun seed supplémentaire ne compense
un pilote négatif. Les artefacts v3/v4 rejetés restent conservés ; le runtime
de production ne change qu’après toutes les gates et revue humaine.
