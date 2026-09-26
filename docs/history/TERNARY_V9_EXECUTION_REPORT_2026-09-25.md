# Rapport d'exécution — Ternary V9.1 Bonsai

Date : 25 septembre 2026  
Périmètre : Stable Audio 3 Medium, cœur attention/FFN ternaire, supports
conservés natifs.  
Statut : pilote technique réussi sur les fenêtres testées ; revue audio
humaine en attente ; aucun modèle complet livré.

## Décision courte

Le plan fonctionne au niveau du pilote, mais il n'est pas honnête de dire
que la ternarisation complète est terminée. G32 est le meilleur compromis
mesuré. [0,1] passe avec deux seeds ; le bloc 2 ne passe qu'après ajout d'un
rollout on-policy de quatre pas. Cette interaction doit être conservée dans
la suite de la cascade et contrôlée à chaque fenêtre.

## Contrat et stockage

Teacher : `dit_medium_f16.npz`, SHA-256
`f9e5647ea3225818657d47d47ae4b34afa29c0568206ca89566c1a758944a38e`.  
Contrat : [`contract-v9.1-final.json`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-p0/contract-v9.1-final.json).

| Élément | Résultat |
|---|---:|
| Matrices cœur | 168 |
| Supports natifs | 357 |
| Fraction éléments cœur | 93,5038 % |
| Payload G128, hors conteneur | 550 196 256 octets |
| Payload G32, hors conteneur | 613 897 248 octets |
| Envelope projet | 650 000 000 octets |

Le cœur est strictement `q ∈ {-1,0,+1}`, avec une scale FP16 par groupe et
`W=s*q`. Les supports ne sont pas ternarisés.

## Gates P0/P1

- Parité dense en processus séparés : sortie `[1,256,128]`, erreur relative
  nulle dans le fixture, cosinus 1.
- Replay du cache : 16 états, longueurs 64/128, huit sigmas, exact.
- Tests : contrat Bonsai, packing, gradient-checkpointing, reprise et RNG
  passent.
- Cache états :
  `output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/state-cache-512-v9/`.
- Cache cibles teacher :
  `output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/teacher-targets-512-v9/targets.npz`.
- Couverture : 512 états, 16 prompts, huit sigmas, 43 parents ; mémoire de
  construction des cibles : pic 3,66 GiB Metal.

## Pilotes full-DiT

Les audits ci-dessous utilisent sept prompts et quatre sigmas. Les valeurs
mean/min sont les cosinus velocity teacher/student ; `release` est le gate
technique interne, pas une note musicale.

| Run | mean/min | release |
|---|---:|---|
| [`G128 [0,1]`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window0-1-g128-112/) | 0,96056 / 0,86240 | PASS |
| [`G128 [1,2]`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window1-2-g128-112/) | 0,95200 / 0,82577 | FAIL |
| [`G32 [0,1] seed 1`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window0-1-g32-112/) | 0,96492 / 0,88516 | PASS |
| [`G32 [0,1] seed 2`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window0-1-g32-112-seed2/) | 0,96193 / 0,86794 | PASS |
| [`G32 bloc 2 pointwise`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-block2-g32-112/) | 0,95662 / 0,84431 | FAIL |
| [`G32 bloc 2 on-policy 4 pas`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-block2-g32-112-onpolicy4/) | 0,95735 / 0,85615 | PASS |

La faiblesse G128 [1,2] est localisée dans le prompt piano et autour de
`sigma=0,890903`. Une tentative on-policy G128 sur deux blocs a dépassé la
garde mémoire à 12,33 Go. La version G32 on-policy d'un seul bloc reste à
7,85 Go de pic observé.

## Canaris audio bruts

Tous les canaris retenus sont finis, stéréo et 44,1 kHz. Ils ne sont pas
normalisés par EQ, compression, limiteur ou autre traitement réparateur.

| Run | Audio cosine | Audio L2 relative | Latent L2 relative |
|---|---:|---:|---:|
| G32 [0,1] seed 1 | 0,96555 | 0,2671 | 0,2023 |
| G32 [0,1] seed 2 | 0,93518 | 0,3762 | 0,2798 |
| G32 bloc 2 on-policy | 0,94665 | 0,3241 | 0,2364 |

Fichiers à écouter :

- [teacher G32 [0,1] seed 1](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window0-1-g32-112/audio-canary-4steps/teacher.wav)
- [student G32 [0,1] seed 1](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window0-1-g32-112/audio-canary-4steps/student.wav)
- [teacher G32 [0,1] seed 2](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window0-1-g32-112-seed2/audio-canary-4steps-128/teacher.wav)
- [student G32 [0,1] seed 2](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-window0-1-g32-112-seed2/audio-canary-4steps-128/student.wav)
- [teacher G32 bloc 2](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-block2-g32-112-onpolicy4/audio-canary-4steps-128/teacher.wav)
- [student G32 bloc 2](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/full-dit-block2-g32-112-onpolicy4/audio-canary-4steps-128/student.wav)

Un canari lancé avec crop 256 a été écarté : il ne correspondait pas à la
longueur 128 des caches et des comparaisons précédentes.

## Verdict et suite autorisée

Verdicts actuels :

- `format_pass` : **non déclaré** — aucun export autonome 168/168 final ;
- `technical_pass` : **pilote oui**, sur les runs ci-dessus ;
- `quality_accepted` : **non** — écoute A/B humaine et test réservé absents.

Ne pas lancer les 24 fenêtres tant que la revue audio n'est pas faite. Après
validation, reprendre avec G32, fenêtre active d'un bloc, on-policy 4 pas
uniquement si le gate de trajectoire l'exige, canary après chaque fenêtre,
rollback au dernier jalon accepté, puis export autonome depuis les records
rechargés.
