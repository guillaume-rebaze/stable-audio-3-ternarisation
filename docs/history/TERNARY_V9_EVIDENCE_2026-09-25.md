# V9 — vérifications et corrections du diagnostic V7/V8

Date : 25 septembre 2026. Statut : audit de préparation, pas entraînement V9.

Le [plan V9](TERNARY_QUALITY_RECOVERY_PLAN_V9.md) s'appuie sur ce registre.
Les anciens résultats sont conservés. Une faiblesse de mesure ne prouve ni
que le modèle est bon, ni qu'elle explique à elle seule l'échec audio.

Mise à jour : les sections 1–8 documentent le diagnostic initial ; la
section 9 chiffre le nouveau périmètre Bonsai demandé par l'utilisateur.
La cible toutes-matrices/500 Mo n'est plus le contrat actif.

## 1. Résultat acquis : aucun modèle complet accepté

Le checkpoint
[`ttq-block0-teacher-anchor-64updates-v1/records_checkpoint.json`](../output/sample-expertise-pilot/ternary-quality-v7-20260925/ttq-block0-teacher-anchor-64updates-v1/records_checkpoint.json)
déclare `quantizer_mode=ttq`, G32 et **7 matrices**. Son payload a le SHA-256
`37b5a9a868b90fd4d0dbf26d40637d24e77474864ebdba713864634be55d6672`.
Ce n'est pas un DiT complet : il ne couvre même pas les deux projections de
conditionnement local du bloc 0.

Sur le benchmark V8 train sélectionné, le record source est à
`0,969460 / 0,895406` en velocity mean/min. Le candidat activation-aware tombe
à `0,866670 / 0,496903`. Sur les 12 prompts du split nommé validation :

| Record | Velocity mean/min | Terminal mean/min |
|---|---:|---:|
| Source TTQ V7 | 0,958326 / 0,871733 | 0,769956 / 0,642287 |
| Activation-aware V8 | 0,847597 / 0,667847 | 0,446067 / 0,285545 |

Sources : [source](../output/sample-expertise-pilot/ternary-quality-v7-20260925/audit-v8-validation-source-v1/audit_summary.json),
[candidat](../output/sample-expertise-pilot/ternary-quality-v7-20260925/audit-v8-validation-activation-aware-max-v1/audit_summary.json).
Les comparaisons appariées de ce round justifient le rejet V8. Les valeurs
historiques V7 `0,96620 / 0,82497` ne constituent pas la même sélection ni le
même seed ; leur comparaison directe avec `0,96946 / 0,89541` serait invalide.

## 2. Symlinks : correction de l'interprétation précédente

Le constructeur
[`build_ternary_independent_corpus.py`](../services/musicgen/build_ternary_independent_corpus.py)
stage explicitement les données par liens symboliques. Le contrôle du
25 septembre a vérifié :

- 434 latents dans le train, tous symlinks ;
- 16 exemples dans le contrat embarqué du cache V7 ;
- **16/16 chemins résolus appartiennent aux cibles du train déclaré** ;
- aucun de ces 16 chemins absent de cet ensemble.

Un chemin résolu vers `universal-dataset` ou `broad-music` ne démontre donc
pas une contamination. V8 corrige une représentation des chemins, pas une
fuite train/test prouvée par cette observation. Cela ne blanchit pas les
anciens audits lancés intentionnellement ou accidentellement sur une autre
sélection ; ce problème distinct reste réel.

Le [contrat V8](../output/sample-expertise-pilot/ternary-quality-v7-20260925/dataset-contract-v8.json)
compte bien `434/33/41` exemples. Son « zéro parent overlap » porte sur les
identifiants enregistrés, parfois issus de noms de fichier, pas sur une
réconciliation exhaustive de la filiation audio ou des PCM décodés.
`prompt_overlap_allowed=true` : contrairement au texte du plan V8, la
disjonction des prompts n'est pas une exigence de ce validateur.

## 3. La preuve de validation n'est pas fermée

Les deux `audit-v8-validation-*/audit_summary.json` portent :

```json
{
  "heldout": false,
  "split": {
    "verified": false,
    "reason": "no split manifest supplied"
  }
}
```

Un autre champ, `dataset_contract.valid`, vaut vrai. C'est une incohérence
entre deux mécanismes de validation, pas une preuve automatique de fuite.
Il faut les unifier avant de présenter un résultat comme test indépendant.

Autres limites observables dans
[`ternary_provenance_v8.py`](../services/musicgen/ternary_provenance_v8.py),
[`audit_ternary_quality.py`](../services/musicgen/audit_ternary_quality.py) et
[`profile_ternary_activations_v8.py`](../services/musicgen/profile_ternary_activations_v8.py) :

- le validateur vérifie les fichiers listés, pas l'égalité avec l'inventaire
  entier du répertoire ; l'auditeur charge ensuite le répertoire ;
- le contrat corpus n'est pas lié obligatoirement au contenu du cache utilisé
  par le profileur ; des hashes sont journalisés, sans comparaison à un
  contrat d'expérience unique ;
- le retour de `validate_contract()` n'inclut pas `dataset_digest` ;
- un ensemble consulté pour choisir seuils, modules et checkpoints est du
  **développement**, même sans aucun gradient calculé dessus.

Les 12 prompts de validation sont vocaux. Ils ne suffisent pas à établir une
qualité globale sur piano, funk, rock, musiques électroniques et ambiances.

## 4. Le profil 256 états ne couvrait pas les trajectoires

Le [manifest du cache](../output/sample-expertise-pilot/ternary-quality-v7-20260925/state-cache-512-fp32-balanced-v7-identity/manifest.json)
décrit 512 états : 256 `real_latent_noised`, puis 256 `teacher_trajectory`.
La lecture effective de `states.npz` confirme ces deux classes et leur ordre.

Le [profil 256](../output/sample-expertise-pilot/ternary-quality-v7-20260925/activation-profile-block0-256-v2/activation_profile.json)
sélectionne exactement les indices 0–255, dans un autre ordre. Résultat :
**256 états réels bruités, 0 trajectoire teacher, 0 trajectoire student**.
Il couvre bien 16 prompts × 8 sigmas × 2 réplicas, mais ces axes ne suffisent
pas à décrire sa couverture de la distribution d'inférence.

Le profileur moyenne les quantiles calculés à chaque appel. Ce n'est ni le
quantile du mélange de tous les tokens, ni nécessairement le quantile des
valeurs absolues. Son wrapper réimplémente aussi `Linear` par matmul puis
addition sans test de parité avec l'opérateur original.

La variation avec sigma justifie une calibration représentative. Elle ne
démontre pas que des poids ou seuils statiques seraient impossibles. V9
conserve des poids statiques et n'ajoute pas huit banques de poids.

## 5. Rejeu dense du cache bloc : écart petit, réel, non causal à lui seul

Le collecteur V8 convertit `h_in`, `context`, `global_cond`, `local_padded` et
`target` en FP16 après le calcul teacher. Le banc les rejoue tous en FP16.
Le runtime original emploie aussi du FP32, notamment sur le chemin temporel.

Un contrôle en lecture seule a chargé les poids denses du **bloc 0 seul**,
avec `strict=True`, puis rejoué quatre entrées du cache existant :

| Index cache | Sigma | Cosinus dense/rejeu | Erreur L2 relative |
|---:|---:|---:|---:|
| 0 | 0,273885 | 0,99999940 | 0,00070166 |
| 7 | 1 | 0,99999934 | 0,00058407 |
| 63 | 1 | 0,99999934 | 0,00060621 |
| 127 | 1 | 0,99999923 | 0,00064471 |

Pic Metal de cette sonde : **147 896 864 octets**. Ce sont quatre contrôles,
pas un audit des 128 états ni un test de production. L'erreur relative mesurée
est inférieure à 0,1 % : elle n'explique pas à elle seule l'effondrement V8.
Un cache de référence V9 doit néanmoins préserver les dtypes et disposer
d'un témoin dense systématique.

Défaut distinct dans
[`benchmark_ternary_block_records_v8.py`](../services/musicgen/benchmark_ternary_block_records_v8.py) :
`T_lat` est tiré de `shape[2]` au lieu de `shape[1]`. Le cache est
`[128, 192, 1536]`, dont 64 memory tokens : la longueur correcte est 128,
pas 1472. Le chemin testé appelle directement le bloc ; ce défaut n'est donc
pas une démonstration que ses anciens scores sont tous faux.

## 6. Contrat ternaire et objectif d'apprentissage

Le TTQ local reconstruit `m + s_pos·1[q=1] − s_neg·1[q=-1]`.
Il ne satisfait pas en général `W=s·q`, avec zéro exact et niveaux opposés.
Le centre de quantification n'est pas le biais vectoriel d'une couche linéaire.
Sources : [contrat](../services/musicgen/ternary_contract.py),
[forward et serialization](../services/musicgen/train_ternary_quality.py).

Autre point à tester, sans le déclarer cause prouvée : `group_means` FP32
n'est pas sauvegardé directement dans le record TTQ ; il est reconstruit à
partir de `biases` et `scales` FP16. V9 strict supprime ce centre à l'inférence.

Le [calibrateur V8](../services/musicgen/calibrate_ternary_activation_aware_v8.py)
minimise une erreur de poids pondérée par `E[x_i²]`. Il ignore les termes
croisés `E[x_i x_j]`, le bloc non linéaire, le suffixe et le sampler.
Il compare aussi un candidat déjà compensé à la reconstruction des poids
denses originaux. Une amélioration de cet objectif peut défaire la
compensation utile. Le rejet mesuré ne condamne pas toute calibration
activation-aware ni tout STE.

Enfin, « pas de flips = pas de progrès » est trop fort : une modification des
scales peut réellement améliorer la fonction à codes identiques. Le seul
verdict de promotion doit venir des sorties rechargées et de l'audio, pas du
nombre de flips, dans un sens comme dans l'autre.

## 7. Taille et ressources revérifiées

Les headers du NPZ dense ont été lus sans charger tout le modèle en mémoire :
525 tableaux, 1 453 368 336 éléments ; 230 tenseurs de dimension ≥2 totalisent
1 452 609 536 éléments. Cela inclut les memory tokens et la projection apprise
`cond.seconds_total_weight` ; un prédicat limité à `.weight` les manquerait.

Hypothèse : deux bits par code, padding par ligne/groupe, une scale FP16 par
groupe, autres tenseurs à leur précision originale. Hors headers du conteneur :

| Groupe | Codes, octets | Scales, octets | Autres, octets | Total théorique, octets |
|---:|---:|---:|---:|---:|
| 16 | 363 290 624 | 181 645 312 | 1 520 672 | 546 456 608 |
| 32 | 363 438 080 | 90 859 520 | 1 520 672 | 455 818 272 |
| 64 | 363 732 992 | 45 466 624 | 1 520 672 | 410 720 288 |
| 128 | 364 322 816 | 22 770 176 | 1 520 672 | 388 613 664 |

Ces chiffres confirment V6, pas l'existence d'un modèle de cette qualité.
Le mélange de **tailles de groupes**, sans mélange de niveaux de poids, peut
allouer une partie des 44,18 MB de marge G32 aux matrices sensibles.

Les champs historiques `metal_*_gb` divisent par `1024**3` : ce sont des GiB.
Le pic du profil V8 est **3 955 470 120 octets**, soit 3,955 GB ou 3,684 GiB,
et non 3,68 GB. RSS, mémoire de l'allocateur et mémoire totale unifiée ne
doivent pas être additionnées aveuglément ni confondues avec une VRAM CUDA.

État local au contrôle : `df -h .` annonce seulement **9,4 GiB disponibles**.
L'ancien chemin `.venv/bin/python` du runtime n'existe pas, mais le Python
local `/Users/guillaumegaillard/.pyenv/versions/3.12.6/bin/python3` charge
NumPy 2.1.3 et MLX 0.31.2 ; Metal est disponible. Le teacher est présent.
Pas besoin de prétendre que tout le runtime a disparu, ni de réinstaller à
l'aveugle. L'environnement doit être verrouillé avant l'entraînement.

SHA-256 revérifiés :

- DiT : `f9e5647ea3225818657d47d47ae4b34afa29c0568206ca89566c1a758944a38e` ;
- `dit_mlx_medium.py` : `12f4e0743f0c00f010f7e6b962e1de921378d1f4b3a3b77e6f106ece15e3ee4a`.

## 8. Limites de cet audit

Au moment de la rédaction initiale de cet audit, aucun entraînement, export
V9, test final réservé ou nouveau rendu audio n'avait été réalisé. Cette
phrase décrit l'état pré-exécution ; la section 11 consigne les runs pilotes
ultérieurs. Aucun fichier de corpus ni poids teacher n'a été modifié.
Le graphe existant du projet n'indexe pas ce pipeline ; l'audit s'appuie sur
les fichiers et artefacts ci-dessus, pas sur des relations de graphe inventées.

Les défauts relevés sont des raisons de corriger le protocole. Ils ne donnent
pas une probabilité mesurée de réussir une conversion intégrale.

## 9. Révision Bonsai : inventaire et stockage réellement applicables

Décision du 25 septembre 2026 : cœur attention/FFN ternaire, supports natifs
conservés, comme compromis de déploiement Bonsai. L'utilisateur confirme
ensuite préférer environ 550–650 Mo au plafond précédent de 500 Mo.

Une nouvelle lecture des **headers uniquement** du NPZ a été effectuée avec
`zipfile` et `numpy.lib.format`, sans charger les poids sur Metal :

```text
/Users/guillaumegaillard/.cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz
```

Taille de l'archive source : 2 907 300 946 octets, 525 tableaux.
La sélection emploie les sept suffixes `CORE_NAMES` de
[`train_ternary_quality.py`](../services/musicgen/train_ternary_quality.py),
avec correspondance exacte `transformer.layers.<index>.<module>.weight` :

| Catégorie | Tableaux | Éléments | Stockage natif des supports |
|---|---:|---:|---:|
| Cœur attention/FFN | 168 | 1 358 954 496 | Sans objet |
| Autres tableaux, supports et buffers | 357 | 94 413 840 | 189 223 968 octets |
| Total | 525 | 1 453 368 336 | — |

Part des éléments du cœur : 93,5037913 %. Il s'agit d'un inventaire de
tableaux, pas d'une nouvelle mesure de paramètres entraînables. Les supports
incluent du FP32, notamment `cond.seconds_total_weight`, et ne doivent pas
être convertis aveuglément en FP16.

Pour chaque matrice du cœur `[out, in]`, calculer
`groupes = out * ceil(in / G)` ; codes = `groupes * G / 4` octets,
scales = `groupes * 2` octets. Ajouter les autres tableaux à leur dtype
natif. Les dimensions du cœur sont divisibles pour les groupes ci-dessous.

| Groupe | Codes | Scales | Supports | Total hors conteneur |
|---:|---:|---:|---:|---:|
| 16 | 339 738 624 | 169 869 312 | 189 223 968 | 698 831 904 |
| 32 | 339 738 624 | 84 934 656 | 189 223 968 | 613 897 248 |
| 64 | 339 738 624 | 42 467 328 | 189 223 968 | 571 429 920 |
| 128 | 339 738 624 | 21 233 664 | 189 223 968 | 550 196 256 |

Ces chiffres supposent une scale FP16 par groupe, pas deux tableaux affines
indépendants. Ils ne sont pas la mesure d'un export existant. La taille
réelle inclura le conteneur et les métadonnées indispensables ; l'encodeur
texte et le codec restent séparés. Les 455 818 272 octets de la section 7
restent corrects **pour l'ancien périmètre toutes-matrices G32 uniquement**.

Le dernier `df -k .` de cette révision rapporte 7 555 288 KiB disponibles,
soit 7 736 614 912 octets (~7,74 GB / 7,21 GiB). Ce point de mesure remplace
le chiffre de disponibilité précédent, pas les mesures historiques de runs.
Il ne prouve pas qu'un entraînement entre dans cette réserve ; le plan exige
un budget de croissance et une réserve système avant lancement.

## 10. Portée de la révision et correction de confiance

Le [livre blanc Bonsai Image](https://github.com/PrismML-Eng/Bonsai-Image-Demo/blob/main/bonsai-image-4b-whitepaper.pdf)
a été consulté, notamment ses sections de format, d'évaluation et l'annexe D.
Il décrit un cœur ternaire et des supports flottants : cette distinction,
confirmée par l'utilisateur, change notre contrat final. L'architecture SA3
n'est pas celle de FLUX ; la correspondance retenue reste une adaptation.

Les documents consultés et le
[dépôt d'inférence](https://github.com/PrismML-Eng/Bonsai-Image-Demo) ne
constituent pas une reproduction de leur entraînement. Le plan distingue
explicitement les résultats publiés, nos mesures et les recettes proposées.

L'estimation 10–30 % est retirée car subjective et non calibrée. Le retrait
des seuils arbitraires de similarité comme verdict musical est décidé avant
les runs V9, pas pour requalifier les échecs V8. Les anciennes V9 et décisions
sont conservées dans l'archive ; aucun nouveau résultat qualité n'est acquis.

La phase documentaire initiale est maintenant suivie d'une exécution pilote
bornée. Aucun poids teacher ni fichier de corpus n'a été modifié ; les
records d'expérimentation et les WAV sont écrits sous `output/`.

## 11. Exécution V9.1 — P0 à P2

### P0 : contrats et banc

Le contrat actualisé est
[`contract-v9.1-final.json`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-p0/contract-v9.1-final.json).
Il a été construit depuis le teacher réel
`dit_medium_f16.npz`, SHA-256
`f9e5647ea3225818657d47d47ae4b34afa29c0568206ca89566c1a758944a38e`.
Il vérifie 168 matrices cœur, 357 supports, 1 358 954 496 éléments cœur,
et les enveloppes hors conteneur G128/G32 de 550 196 256 / 613 897 248
octets. Le contrat est indépendant de MLX et refuse le code réservé de
packing ; les tests contractuels passent.

Les rapports P0 sont dans
`output/sample-expertise-pilot/ternary-quality-v9-bonsai-p0/` :

- `dense-parity.json` : deux processus frais, sortie `[1,256,128]`, erreur
  relative 0, cosinus 1, passage ;
- `cache-replay.json` : 16 états, longueurs 64/128 et huit sigmas, rejeu
  exact ;
- les tests de reprise/RNG, gradient-checkpointing et packing passent.

Le premier run legacy du bloc 0 a néanmoins écrit un export dense d'environ
2,59 Go avant que le mode `--pilot-only` soit ajouté à
`train_ternary_quality.py`. Les runs suivants ont utilisé des
`records_checkpoint.npz` sans reproduire cette croissance ; l'espace disque
reste une ressource à surveiller et aucun nettoyage destructif n'a été fait.

### P1 : données consommées

Le cache réellement utilisé est
`output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/state-cache-512-v9/`.
Il contient 512 états : 256 `real_latent_noised` et 256 états de trajectoire
teacher, 16 prompts, huit sigmas et 43 parents. Les cibles teacher séparées
sont dans `teacher-targets-512-v9/targets.npz`, forme `[512,256,128]`,
FP16. Le pic de création de ces cibles est 3,66 GiB Metal.

Pendant la préparation du cache on-policy, un bug réel de reproductibilité a
été trouvé dans `prepare_ternary_rollout_targets.py` : la formule de seed
référençait `repeats` hors de sa portée. Elle a été remplacée par le nombre
effectif maximal de répétitions, puis le cache a été reconstruit et ses 32
rollouts teacher vérifiés avant l'entraînement du bloc 2.

### P2 : expériences et résultats

Tous les chiffres suivants proviennent de l'audit DiT complet sur sept
prompts et quatre sigmas, et non d'une seule loss de bloc.

| Run | mean/min velocity cosine | Résultat |
|---|---:|---|
| `full-dit-window0-1-g128-112` | 0,96056 / 0,86240 | release pass |
| `full-dit-window1-2-g128-112` | 0,95200 / 0,82577 | release fail ; cas piano / sigma 0,8909 |
| `full-dit-window0-1-g32-112` | 0,96492 / 0,88516 | release pass |
| `full-dit-window0-1-g32-112-seed2` | 0,96193 / 0,86794 | release pass |
| `full-dit-block2-g32-112` | 0,95662 / 0,84431 | release fail |
| `full-dit-block2-g32-112-onpolicy4` | 0,95735 / 0,85615 | release pass |

Le rollout on-policy 4 pas du bloc 2 a culminé à 7,85 Go sous la garde
11 Go. Une tentative G128 sur deux blocs a dépassé 12,33 Go et a été
interrompue par la garde ; le dépassement n'a pas été contourné.

Les canaris audio cohérents, tous finis et à 44,1 kHz, donnent :

- G32 [0,1] seed 1 : audio cos 0,96555, L2 relative 0,2671, latent L2
  0,2023 ;
- G32 [0,1] seed 2 : audio cos 0,93518, L2 relative 0,3762, latent L2
  0,2798 ;
- G32 bloc 2 on-policy : audio cos 0,94665, L2 relative 0,3241, latent L2
  0,2364.

Le premier canari du bloc 2 lancé avec crop 256 a été écarté comme mesure
non comparable ; le canari retenu emploie crop 128, comme le cache et les
résultats précédents.

### Ce que la preuve établit — et ce qu'elle n'établit pas

Elle établit la faisabilité d'un pilote full-DiT strict G32 sur les fenêtres
testées, avec supports natifs, reload exact et adaptation on-policy utile
pour le bloc 2. Elle n'établit pas encore une cascade complète : aucun
checkpoint 24 blocs, export autonome Bonsai 168/168, test réservé ni écoute
A/B humaine n'est disponible. Le statut correct reste
`technical_pilot_pass` / `musical_review_pending`, pas `quality_accepted`.

Le premier échec observé dans cette campagne est donc localisé : la
quantification pointwise du bloc 2 ne respecte pas le minimum release ; les
états student de production corrigent ce cas dans le run on-policy, au prix
d'un coût mémoire et d'une recette spécifique. P3 reste suspendu jusqu'à
la revue audio et au gel de cette recette.
