# Run ternaire v4 — corpus autorisé — 22 septembre 2026

Note du 23 septembre : résultats numériques conservés ; interprétation corrigée.
Le split est séparé par identifiants de noms, sans preuve complète de séparation
des œuvres/PCM. Son test a été consulté ; il devient du développement historique.
La suite est définie dans le [plan v6](TERNARY_QUALITY_RECOVERY_PLAN.md).

## Verdict

**Rejeté. Expérimental seulement.**

L’autorisation utilisateur couvre l’étude personnelle locale, pas la
redistribution. Les sidecars d’origine restent inchangés et leurs droits
natifs restent `provided_unreviewed`/non déclarés.

## Données et split

Le builder `services/musicgen/build_ternary_independent_corpus.py` compare les
parents via `src_relpath`/`path`, conserve les dérivés d’un même parent dans un
seul split et exclut les sources déjà présentes dans le train.

| Split | Samples | Parents | Prompts |
|---|---:|---:|---:|
| Train | 253 | 202 | 28 |
| Validation | 39 | 28 | 18 |
| Test | 79 | 66 | 24 |

46 sources broad recouvrant le train v3 par nom ont été exclues du held-out.
Le préflight indiquait `independent_available=true`, sans chevauchement de ses
identifiants de parent. Ce booléen ne prouve pas l'absence de doublons audio,
de sources d'une même œuvre ou d'exposition lors des expériences antérieures.

Preuves :

- config : `configs/ternary_quality_v4_authorized.json` ;
- sélection : `output/sample-expertise-pilot/ternary-quality-v3/g2-authorized-expanded-v3/selection_manifest.json` ;
- préflight : `output/sample-expertise-pilot/ternary-quality-v3/g2-authorized-expanded-v3/preflight-final/`.

## Entraînement

- DiT Medium, scope 168, strict symétrique `W_hat=s*q` ;
- G64, `q ∈ {-1,0,+1}`, `24 × 200` updates, polish 200 ;
- checkpoints complets toutes les 50 étapes ;
- run/checkpoints : `/Volumes/Extreme SSD/OnUsLoopLab/ternary-quality-v4-authorized-20260922/g64-expanded/`.

Export : `479 660 087` octets. Reload : parité paramètres exacte (`858/858`),
erreur relative `0`, cosinus `1`.

## Résultats

| Gate | Résultat |
|---|---|
| Validation 18 prompts × 5 sigmas | cosine `0,81248 / 0,65934`, rejeté |
| Test 24 prompts × 5 sigmas | cosine `0,80827 / 0,69551`, rejeté |
| Audio brut, 3 prompts validation | `0/3` pass ; RMS `1,610`, `1,462`, `1,484` |
| Metal entraînement | pic `5,42 GiB` |
| Reload | Metal `0,57 GiB`; RSS `1 409 941 504` octets; swap `7 588,19 MiB` |

Le bloc 23 conserve une erreur dense moyenne `0,48034`. C'est une distance
entre poids maîtres et poids quantifiés, pas une borne de capacité ni une
mesure audio. Ce run n'a pas atteint les critères de qualité ; il n'isole pas
l'effet de la quantité de données. Ne pas promouvoir l'artefact.

## Suite révisée

Convertir les poids préentraînés selon le plan v6 : quantizer appris,
récupération globale revisitant tous les blocs et extension du scope.
Garder v4 comme contrôle rejeté ; réserver un nouveau test final. Ni un
entraînement depuis zéro ni une impossibilité du ternaire ne sont établis ici.
