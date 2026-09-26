# Registre de preuves — récupération ternaire, 22 septembre 2026

Statut : aucun modèle musicalement validé. Ce registre décrit les essais effectués ; les expériences futures sont dans le [plan v3](TERNARY_QUALITY_RECOVERY_PLAN.md).

## 1. Portée des résultats historiques

Racine des sorties : `output/sample-expertise-pilot/ternary-quality-recovery/`.

L’audit velocity emploie huit exemples/prompts du corpus d’entraînement et cinq sigmas `0.95/0.75/0.50/0.25/0.10`, soit 40 cas. Ce n’est pas un test indépendant de généralisation. Les mesures audio ci-dessous concernent le premier prompt piano, seed 42, 12 pas, 128 tokens latents, canal gauche seulement. Durée nominale « 12 s », environ 11,889 s avec le facteur codec local.

Le cosinus spectral historique provient de `log1p(PSD)` aplati. Les valeurs sont conservées, mais le seuil 0,85 n’est pas un seuil perceptuel validé. Aucun résultat d’écoute comparée n’est associé à ces runs.

| Run | Velocity moyenne / minimum | Corrélation spectrale historique | RMS student / teacher |
|---|---:|---:|---:|
| group64-core | 0,807602 / 0,678008 | 0,161816 | 1,169820 |
| group32-core | 0,847508 / 0,712991 | 0,240763 | 1,612681 |
| fp16-b23 | 0,852821 / 0,718317 | non mesurée ici | non mesuré ici |
| group32-window20-23 | 0,850208 / 0,722306 | non mesurée ici | non mesuré ici |
| group32-window16-19-hi | 0,800171 / 0,647969 | non mesurée ici | non mesuré ici |
| group32-rollout20-23 | 0,812680 / 0,692359 | 0,227914 | 0,869165 |
| group32-adapter-r8 | pas de JSON velocity retrouvé pour ce run | 0,154800 | 0,794261 |
| group32-adapter-real | 0,868208 / 0,747619 | 0,151456 | 1,501435 |

Sources directes :

- [group64 : velocity](../output/sample-expertise-pilot/ternary-quality-recovery/group64-core/velocity_metrics.json), [audio](../output/sample-expertise-pilot/ternary-quality-recovery/group64-core/audio-gate-12s/audio_metrics.json), [run](../output/sample-expertise-pilot/ternary-quality-recovery/group64-core/run_summary.json).
- [group32 : velocity](../output/sample-expertise-pilot/ternary-quality-recovery/group32-core/velocity_metrics.json), [audio](../output/sample-expertise-pilot/ternary-quality-recovery/group32-core/audio-gate-12s/audio_metrics.json), [run](../output/sample-expertise-pilot/ternary-quality-recovery/group32-core/run_summary.json).
- [bloc 23 FP16](../output/sample-expertise-pilot/ternary-quality-recovery/fp16-b23/velocity_metrics.json).
- [fenêtre 20–23](../output/sample-expertise-pilot/ternary-quality-recovery/group32-window20-23/velocity_metrics.json), [fenêtre 16–19 LR élevée](../output/sample-expertise-pilot/ternary-quality-recovery/group32-window16-19-hi/velocity_metrics.json).
- [rollout teacher : velocity](../output/sample-expertise-pilot/ternary-quality-recovery/group32-rollout20-23/velocity_metrics.json), [audio](../output/sample-expertise-pilot/ternary-quality-recovery/group32-rollout20-23/audio-gate-12s/audio_metrics.json).
- [adapter teacher rollout : audio](../output/sample-expertise-pilot/ternary-quality-recovery/group32-adapter-r8/audio-gate-12s/audio_metrics.json).
- [adapter latents réels : velocity](../output/sample-expertise-pilot/ternary-quality-recovery/group32-adapter-real/velocity_metrics.json), [audio](../output/sample-expertise-pilot/ternary-quality-recovery/group32-adapter-real/audio-gate-12s/audio_metrics.json).

La valeur velocity `0,8442/0,7367` de l’adapter rollout avait été rapportée dans le compte rendu précédent ; absence de JSON dédié retrouvé, donc ne pas la confondre avec les mesures archivées ci-dessus.

## 2. Configuration et comparabilité

- Group64 : 100 steps/bloc et 100 de polish ; durée 1 628,28 s.
- Group32 : 200 steps/bloc et 200 de polish ; durée 3 205,37 s. Ajout d’une loss velocity terminale par rapport au premier run group64.
- Les deux utilisent 193 latents, huit conditionnements, crop 128 et seed 42. Les fichiers `config.json` accompagnent les runs.
- Fenêtres : QAT réouverte depuis poids déquantifiés ; historique des poids maîtres et optimiseur non préservé. Pas d’essai zéro-step établissant l’absence de dérive lors de cette réouverture.
- Variante rollout : états teacher mis en cache à quatre pas, pas supervision multi-pas différentiable ni états student actualisés.
- Adapters rank8 : 300 updates ; poids `up/down` FP32 dans le prototype, pas FP16. Ils rendent le modèle hybride.

Conclusion : group32 n’est pas une ablation isolée de granularité ; l’entraînement et l’objectif changent aussi. Ces résultats ne prouvent pas qu’un groupe 16 ou un rang 32 résoudrait l’échec.

## 3. Reload : fidélité non clôturée

| Run | Erreur relative max | Cosinus min | Cas contrôlés |
|---|---:|---:|---:|
| group64-core | 0,002152928 | 0,999998033 | 3 |
| group32-core | 0,002489156 | 0,999997258 | 3 |

Ces erreurs dépassent l’ancien objectif `0.001`. Le trainer accepte maintenant `0.02` et cosinus `0.9999`, puis émet un avertissement. L’attribution de l’écart à un simple arrondi dense/quantifié n’est pas démontrée.

Constats dans [train_ternary_quality.py](../services/musicgen/train_ternary_quality.py) :

- `hard_freeze_block()` ajoute littéralement `0.0` à la liste des erreurs ; tous les zéros `max_dense_reconstruction_error` des anciens rapports sont non probants.
- Le quantizer QAT travaille avec moyennes/scales FP32 ; le packer utilise des métadonnées FP16.
- L’export caste globalement les paramètres non packés en FP16.
- Les embeddings locaux de la cascade proviennent du teacher et de zéros FP16, alors que le forward complet construit ses zéros locaux sans dtype explicite.

Le [contrat actuel](../services/musicgen/ternary_contract.py) représente `m+s*q`. Les codes sont bien ternaires, mais les niveaux ne sont généralement pas symétriques autour de zéro.

## 4. Sonde des fréquences temporelles, sans entraînement

Le runtime local `models/defs/dit_mlx_medium.py`, classe `ExpoFourierFeatures`, construit 128 fréquences FP32 entre `0.5*2*pi` et `10000*2*pi`. Le checkpoint teacher ne contient pas la clé `timestep_features.freqs` ; le buffer est donc généré par le constructeur. Les exports group32/group64 ajoutent cette clé en FP16.

Sonde exécutée le 22 septembre : recomposition des fréquences avec la formule MLX du constructeur, lecture de la seule petite matrice dans l’export group32, comparaison des features sin/cos. Aucun chargement complet du DiT, aucune modification des poids.

```json
{
  "runtime_dtype": "float32",
  "saved_dtype": "float16",
  "max_abs_freq_delta": 15.8671875,
  "exact_fp16_cast": true,
  "fourier_features_relative_error": {
    "0.95": 0.8004156351089478,
    "0.50": 0.6242261528968811,
    "0.10": 0.3969908654689789
  }
}
```

Reproduction en lecture seule depuis la racine du dépôt :

```bash
rtk proxy python3 - <<'PY'
import math
import numpy as np
import mlx.core as mx

path = "output/sample-expertise-pilot/ternary-quality-recovery/group32-core/dit_medium_ternary_quality_group32_core.npz"
freqs = mx.exp(
    mx.linspace(0.0, 1.0, 128)
    * (math.log(10000.0) - math.log(0.5)) + math.log(0.5)
) * 2 * math.pi
with np.load(path) as archive:
    saved = mx.array(archive["timestep_features.freqs"])
for sigma in (0.95, 0.50, 0.10):
    t = mx.array([sigma], dtype=mx.float16)[:, None]
    a = np.array(mx.concatenate([mx.cos(t * freqs), mx.sin(t * freqs)], axis=-1))
    b = np.array(mx.concatenate([mx.cos(t * saved), mx.sin(t * saved)], axis=-1))
    print(sigma, float(np.linalg.norm(a - b) / np.linalg.norm(a)))
PY
```

Interprétation : conversion non neutre confirmée. L’impact sur la velocity, le rollout et l’écoute reste à isoler en restaurant seulement ce buffer dans un nouvel artefact. Ne pas présenter ces erreurs de features comme des erreurs audio ni comme la cause unique.

## 5. Taille disque et tenseurs : nouveau relevé

Mesure en lecture seule des tailles physiques et des en-têtes NPY contenus dans les archives. Les octets de tenseurs ne comprennent pas tous les surcoûts du runtime.

| Composante | Group64 | Group32 |
|---|---:|---:|
| Fichier NPZ, octets | 512 629 358 | 580 269 830 |
| Codes uint32, octets | 339 738 624 | 339 738 624 |
| Scales + biais de quantification, octets | 84 934 656 | 169 869 312 |
| Autres tenseurs, octets | 188 431 616 | 188 431 616 |
| Total tenseurs décompressés, octets | 613 104 896 | 698 039 552 |

Les codes correspondent à 1 358 954 496 valeurs dans le scope core. Le reste n’est pas négligeable. Le NPZ est plus petit que les tenseurs décompressés ; les valeurs 512/580 Mo ne sont donc pas la mémoire des poids chargés.

Teacher `dit_medium_f16.npz` : fichier 2 907 300 946 octets ; tenseurs 2 907 132 960 octets selon les en-têtes. Les différences de buffers/clefs entre teacher et student interdisent d’assimiler naïvement le nombre d’éléments stockés à un nombre de paramètres appris strictement identique.

Coût de représentation hors padding/conteneur :

- affine 2-bit actuel, scale FP16 + biais FP16 par groupe : `2 + 32/G` bits/poids ; G64 = 2,5, G32 = 3 ;
- symétrique avec une seule scale FP16 stockée : `2 + 16/G` ; G64 = 2,25, G128 = 2,125 ;
- si le backend requiert aussi le biais affine dérivé, son coût mémoire existe même si le fichier ne le stocke qu’une fois implicitement ;
- `log2(3)` est une information théorique, pas le coût du packing actuel.

Adapters : [rollout](../output/sample-expertise-pilot/ternary-quality-recovery/group32-adapter-r8.log) 34 022 172 octets ; [latents réels](../output/sample-expertise-pilot/ternary-quality-recovery/group32-adapter-real.log) 34 011 584 octets. Ajouter ces tailles à la base ; ne pas annoncer l’adapter seul comme taille du modèle.

## 6. Mémoire observée, et non prouvée

Les scripts divisent les compteurs Metal par `1024**3`, malgré des clés suffixées `_gb`. Les nombres suivants sont des **GiB**, pas des GB décimaux :

- Group64 entraînement : 5,5066 GiB ; group32 : 5,7044 GiB.
- Fenêtre 20–23 : 7,7849 GiB ; variante rollout : 7,7630 GiB.
- Adapter rollout : 5,9497 GiB ; adapter réel : 5,9093 GiB.
- QAT globale, un pas : 29,4195 GiB, [log](../output/sample-expertise-pilot/ternary-quality-recovery/gate-e2e.log).
- Génération/décodage group32 : 6,7176 GiB dans le rapport audio ; profil différent de l’entraînement.

Pas de relevé RSS/swap complet dans ces rapports. Ne pas conclure à une conformité système sous 12 GB ni à une impossibilité algorithmique générale à partir de ces seuls compteurs.

## 7. Conclusions conservées et corrections

Conserver : scope manifeste, préfixe réellement exécuté, raw sans normalisation, audits après reload et petits pilotes. Ce sont des fondations utiles, pas des preuves suffisantes.

Retirer des conclusions actives : « reload exact », « mémoire fermée », « problème forcément extérieur au packing », « group16/codebook est la seule voie », « rang supérieur + DAgger fonctionnera », « corrélation spectrale seule certifie la qualité ».

Prochaine expérience informative : corriger/ablationner buffers et contrat sans nouvelle QAT, puis établir une validation indépendante. L’entraînement ne reprend qu’après les gates du plan v3.

## 8. Revalidation G1 post-run

Le loader d’audit valide maintenant les arrays NPZ avant instanciation MLX :
packing `uint32`, codes `{-1,0,+1}`, formes des scales/biais, dérivation du
biais en compact symétrique, signes et tenseur déquantifié. Les 12 tests
ternaires ciblés passent.

Le meilleur snapshot `window-terminal-g64-safe` passe ce contrat sur `168`
tenseurs ; l’audit court reste toutefois rejeté à `0,84439 / 0,80644`
(cosinus moyen/pire). Le contrôle d’intégrité est donc confirmé comme
nécessaire, mais il ne transforme pas le candidat en modèle de qualité.

## 9. G2 autorisée et run v4

Après autorisation explicite pour étude personnelle, les sources locales ont été
réconciliées par parent. `46` sources broad recouvraient le train v3 et ont été
exclues du held-out ; `132` sources non recouvrantes ont été réparties sans
parent partagé :

| Split | Samples | Parents | Prompts |
|---|---:|---:|---:|
| train v4 | 253 | 202 | 28 |
| validation | 39 | 28 | 18 |
| test | 79 | 66 | 24 |

Le run v4 G64 strict symétrique a exporté `479 660 087` octets, avec parité
reload exacte. Les audits held-out restent négatifs : validation
`0,81248 / 0,65934`, test `0,80827 / 0,69551` (moyenne/minimum). Le rendu audio
brut validation échoue sur `3/3` prompts, avec RMS ratios `1,610`, `1,462` et
`1,484`. Ressources : Metal peak entraînement `5,42 GiB`, RSS max
`1 409 941 504` octets, swap `7 588,19 MiB`.

La qualité n’est donc pas limitée uniquement par l’autorisation ou la taille du
corpus : l’hypothèse « plus de données + même core ternaire » est rejetée.
