# Exécution V7 — P2 corrigé et ablation P3

Date : 24 septembre 2026. Périmètre : Stable Audio 3 Medium, MLX, pilote
des blocs 0–1. Résultat : **échec des gates qualité ; aucun modèle final livré**.

Plan exécuté : [TERNARY_QUALITY_RECOVERY_PLAN.md](TERNARY_QUALITY_RECOVERY_PLAN.md).
Diagnostic antérieur : [TERNARY_V7_FORENSIC_REPORT_2026-09-24.md](TERNARY_V7_FORENSIC_REPORT_2026-09-24.md).

## Décision

Le cache v3 a corrigé l’incohérence de RNG, mais le P2 répété reste hors gate.
L’ablation P3 deux pas ne récupère ni la fidélité de vélocité ni le niveau
audio ; son gain de NMSE terminale est trop faible. **Ne pas passer à 0–3 ni
ternariser les 24 blocs.**

Les essais n’établissent pas que la ternarisation est impossible. Ils
établissent que cette exécution ne produit pas de modèle ternaire de qualité.

## P2 sur cache/runtime corrigés

Cache :
[`state-cache-512-fp32-balanced-v3-runtime-rng-fix1`](../output/sample-expertise-pilot/ternary-quality-v7-20260924/state-cache-512-fp32-balanced-v3-runtime-rng-fix1/manifest.json).
Manifeste : 512 états, 16 prompts train, 8 pas, 2 flux de bruit correctement
séparés (bruit initial `seed`, réinjections `seed+1`). Cibles professeur
FP16 présentes pour 512/512 états. Pic Metal cache/cibles : 3,665 Go.

Répétition P2 : blocs 0–1, G32 symmetric, LR `1e-5 → 1e-6`, 50 updates,
16 états fixes, accumulation 4, seed `20260924`. Loss `0,20842 → 0,09351` ;
331 416 codes sur 113 246 208 changés depuis l’initialisation (0,293 %) ;
pic Metal 6,010 Go. Round-trip records-only exact sur trois fixtures.

Audit développement corrigé, 12 prompts × 8 sigmas : cosine velocity moyen
**0,93262**, minimum **0,78394** (minimum requis 0,80). Cosine terminal
moyen **0,82143**. RMS latent student/teacher croît surtout après
`σ=0,7455` : `1,031` à 0,7455, `1,204` à 0,5125, `1,429` à 0,2739,
**1,534 au terminal**.

Rendu brut, trois prompts de développement, 8 pas, mêmes graines, aucun gain
ou post-traitement : RMS ratios **2,002 / 1,593 / 1,796** ; peak ratios
**1,937 / 1,770 / 1,633**. Échec 3/3.

## Contrôle A et P3 appariés

Cache de paires : 16 prompts train × 2 seeds × 7 transitions adjacentes,
soit 224 paires, 32 par transition. Ancres issues du rollout du candidat P2 ;
targets professeur précalculées depuis les mêmes ancres et réinjections.
Élève puis professeur chargés séquentiellement ; pic Metal 3,665 Go.
Cache :
[`pairs.npz`](../output/sample-expertise-pilot/ternary-quality-v7-20260924/trajectory-pairs-p2-v3-rngfix-seed20260924/pairs.npz).

Même checkpoint source, même cache, mêmes seeds, scope, LR, 50 updates,
accumulation 4. A utilise seulement la loss velocity ; P3 alterne moitié
loss ponctuelle et moitié unroll étudiant différentiable sur deux pas,
`Lv + 0,25 × NMSE_endpoint`.

| Mesure développement | Contrôle A | P3 | Gate |
| --- | ---: | ---: | ---: |
| Cosine velocity moyen | 0,93626 | 0,93633 | ≥ 0,90 — passe |
| Cosine velocity minimum | 0,79239 | 0,79247 | ≥ 0,80 — **échec** |
| Cosine terminal moyen | 0,83450 | 0,83519 | ≥ 0,90 — **échec** |
| NMSE terminale moyenne | 0,74777 | 0,73714 | P3 doit gagner ≥10 % — **gain 1,42 %** |
| RMS latent terminal moyen | 1,4750 | 1,4689 | diagnostic — trop amplifié |
| Codes durs changés après run | 0 | 0 | transition effective — **échec** |
| Pic Metal | 6,435 Go | 6,917 Go | ≤ 11 Go — passe |

Pire NMSE terminale : 1,7863 (A), 1,7512 (P3), sans régression P3 ; ce seul
critère ne compense pas le gain moyen inférieur au seuil. P3 loss finale
0,22612 contre 0,14411 au premier update ; dernier objectif paire 0,25366.
Seuls 132/224 identifiants de paires ont été vus sur 200 tirages, avec
couverture équilibrée des sept transitions (28–29 tirages chacune).

| Prompt | RMS A | RMS P3 | Peak A | Peak P3 |
| --- | ---: | ---: | ---: | ---: |
| disco | 1,939 | 1,932 | 1,880 | 1,874 |
| acid | 1,546 | 1,538 | 1,803 | 1,798 |
| house | 1,708 | 1,700 | 1,533 | 1,525 |

Les six paires audio ratent RMS `[0,70; 1,30]` et peak `[0,50; 1,50]`.
Spectres et enveloppes restent presque inchangés entre A et P3. Les WAV sont
bruts, aucune normalisation/limitation ; pic Metal rendu 8,604 Go.

## Diagnostic et limites

1. L’erreur RNG du cache v2 n’était pas l’unique cause : le P2 v3 échoue aussi.
2. A et P3 n’ont changé aucun code ternaire dur par rapport au checkpoint source.
   Leurs quantiles de scales restent presque identiques. L’audio n’a donc pas
   reçu de nouvelle structure ternaire ; P3 n’a changé le rendu que de moins
   de 1 % sur les ratios mesurés.
3. **Hypothèse prioritaire à vérifier :** les runs A/P3 ont redémarré depuis
   `records_checkpoint.npz`. `quantized_linear_to_qat()` reconstruit alors le
   maître FP32 par déquantification des codes/scales arrondis et initialise un
   nouvel optimiseur. Le résidu FP32 et l’état d’optimiseur du P2 sont perdus.
   C’est cohérent avec 0 transition de code dans A/P3 contre 331 416 pendant
   le P2 initial ; cela ne prouve pas que cette seule perte explique les
   ratios audio proches de 2.
4. La couverture pair complète n’a pas été vue pendant ces 50 updates ; la
   moitié trajectory n’a donc pas fait une époque des 224 paires.

Pas d’écoute humaine formelle : les critères techniques sont déjà rouges.
Split test réservé jamais ouvert. Aucun scope 0–3, aucun modèle 24 blocs,
aucun artefact final ≤500 000 000 octets, aucune acceptation utilisateur.

## Vérifications logicielles

- `py_compile` trainer, constructeur de paires et test : passe.
- Tests V7 ciblés (`test_ternary_v7_trajectory.py` et
  `test_ternary_v7_sampling.py`) : **9 passent**.
- `git diff --check` : passe.
- Suite complète `services/musicgen/tests` : collecte bloquée par
  `ModuleNotFoundError: No module named 'models.defs'; 'models' is not a package`
  lorsque `services/musicgen/models.py` masque le package runtime `models`.
  Aucun résultat E2E n’est déclaré.

## Section historique — proposition P3.1 avant exécution

Ne pas ouvrir P4. Le checkpoint source est présent :
[`window_latest.npz`](../output/sample-expertise-pilot/ternary-quality-v7-20260924/micro-overfit-0-1-g32-symmetric-lr1e-5-seed20260924-v3-runtime-rng-fix1/checkpoints/window_latest.npz)
(1,2 Go), avec son
[sidecar JSON](../output/sample-expertise-pilot/ternary-quality-v7-20260924/micro-overfit-0-1-g32-symmetric-lr1e-5-seed20260924-v3-runtime-rng-fix1/checkpoints/window_latest.json).
Inspection non destructive : schéma v4 complet, `step_next=50`, fenêtre 0–1,
G32 symmetric ; matrices maîtres FP32, moments Adam FP32, pas Adam 50 et état
RNG MLX. Le taux sauvegardé est `1,0089e-6`. Le sidecar lie le payload par
SHA-256.

**Blocage à lever avant le run :** `--resume-step-checkpoint` refuse, à juste
titre, tout `run_signature` différent. P3.1 change l'objectif et le cache ; le
checkpoint P2 ne peut donc pas être passé directement à A/P3. Implémenter un
mode distinct de warm-start avec provenance explicite, sans affaiblir la
reprise exacte existante. Préflight sans update : vérifier schéma/complétude/
SHA-256, scope et formes, charger maîtres + moments, puis prouver que la
projection ternaire reproduit les records P2 et que le forward dur restauré
reproduit les fixtures P2 dans la tolérance. Vérifier aussi au smoke que la
politique de LR reprend effectivement le taux annoncé ; sinon, arrêter et
corriger avant tout entraînement.

Si le préflight passe, créer A et P3 dans deux répertoires neufs depuis le
même checkpoint immuable : mêmes maîtres, moments, RNG/ordre d'échantillons,
cache, scope et politique LR ; seule la loss de trajectoire diffère. Garder
le LR sauvegardé fixe pour les deux bras et journaliser sa valeur réelle à
chaque update. Pour ne pas répéter le défaut de couverture, parcourir les
224 paires du cache sans remise : à deux microbatches trajectoire par update,
112 updates couvrent une fois chaque paire ; A garde le même ordre d'ancres et
le même nombre d'updates/microbatches, avec la loss ponctuelle sur les ancres
où P3 déroule deux pas. Auditer à
mi-parcours et en fin, codes persistants, distances aux seuils, scales,
coverage, métriques de trajectoire et audio brut.

Décision historique : conserver les gates P3 ci-dessus sans assouplissement. Si le
préflight échoue, zéro entraînement. Si les codes restent immobiles ou si
une gate qualité/audio échoue, arrêter avant P4 et réviser le quantizer ou
l'objectif à partir des diagnostics ; pas d'élargissement 0–3. Si toutes les
gates passent, confirmer sur une seconde seed avant toute extension. Cette
section décrit la proposition avant son exécution; les résultats réels sont
consignés dans l'addendum ci-dessous.

Le `.tmp.npz` de 273 Mo laissé par l’interruption d’export utilisateur est
conservé comme trace incomplète ; export P3 vérifié est séparé et son
rechargement est exact.

## Addendum — P3.1 réellement exécuté

P3.1 a été exécuté après préflight, avec le checkpoint FP32/Adam P2, LR fixe
`1,008879735e-6`, `paired_epoch`, 112 updates et couverture `224/224`. Le
préflight a vérifié 0 mismatch de codes, scales et biais sur 113 246 208 codes;
le round-trip de l’artifact est passé. Pic Metal : 6,917 Go.

Le premier audit lancé sur `universal-dataset/latents-12s` est invalide pour la
comparaison : ce n’est pas le corpus du cache. Il est conservé mais exclu. Les
chiffres autoritatifs ci-dessous utilisent
`output/sample-expertise-pilot/ternary-quality-v6-20260923/authorized-independent-sftvoices-v1/train`,
le même corpus que P2/A/P3 :

| Mesure | Contrôle A warm-start | P3.1 warm-start | Gate |
|---|---:|---:|---:|
| Velocity cosine moyen | 0,92804 | 0,92819 | ≥ 0,90 — passe |
| Velocity cosine minimum | 0,73204 | 0,73154 | ≥ 0,80 — **échec** |
| Terminal state cosine moyen | 0,75743 | 0,75873 | ≥ 0,90 — **échec** |
| Terminal state cosine minimum | 0,53015 | 0,52822 | diagnostic rouge |
| Teacher-on-student velocity moyen | 0,95502 | 0,95518 | diagnostic |
| Codes changés depuis le début | 68 241 (0,0603 %) | 69 873 (0,0617 %) | transition utile — **échec** |
| Audio brut RMS `dub/voice/piano` | 1,483 / 1,558 / 1,342 | 1,463 / 1,534 / 1,326 | `[0,70;1,30]` — **échec** |
| Pic Metal entraînement | 6,435 Go | 6,917 Go | ≤ 11 Go — passe |

P3.1 ne corrige donc pas le défaut : gain moyen `+0,015 %` contre A, minimum
un peu inférieur, terminal à peine meilleur. La cause « maîtres FP32 perdus »
était réelle dans l’ancien A/P3 mais n’explique pas l’échec actuel. Le
quantizer symmetric et l’objectif restent insuffisants; A/P3.1 ont dérivé par
rapport au P2 au lieu de sélectionner un meilleur état.

Un pilote `learned_symmetric` de 16 updates démarré depuis les records ternaires
est également rouge (`0,91795 / 0,71337`, terminal `0,70638`, pic 8,94 Go).
Il ne permet pas de conclure contre un quantizer appris démarré depuis le
professeur dense, mais il interdit de masquer une mauvaise projection initiale
par une courte reprise sur records.

**Décision finale du rapport :** fermer P4 et toute cascade 24 blocs. Le plan
de reprise est désormais
[`TERNARY_QUALITY_RECOVERY_PLAN_V7.md`](TERNARY_QUALITY_RECOVERY_PLAN_V7.md) :
calibration comportementale, TTQ/LSQ avec seuils et niveaux appris, transition
progressive, validation indépendante et sélection `best_validation` avant
toute extension de scope.
