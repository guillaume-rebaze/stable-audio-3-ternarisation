# Exécution du plan ternaire v3 — 22 septembre 2026

## Verdict

**Rejeté pour livraison. Expérimental seulement.**

Ce rapport conserve le verdict historique v3. La continuation v4 avec corpus
personnel autorisé et split held-out est documentée à la fin ; elle est elle
aussi rejetée sur la qualité.

Le run a bien exécuté la cascade stricte symétrique `W_hat = s_g * q`, avec
`q ∈ {-1, 0, +1}`, 24 blocs, 200 updates par bloc, checkpoint, export compact
et reload exact. Il ne passe pas la qualité minimale :

- velocity debug train-seen : cosine moyen **0,8304**, pire **0,7184** ;
- audio brut : **2/3** rendus hors gate RMS/peak ;
- aucun split indépendant held-out disponible ;
- écoute humaine non réalisée ;
- swap système déjà à **10,44 GB** pendant le profil final : le pic Metal seul
  ne prouve pas une exécution saine sous 12 GB.

Ne pas utiliser cet artefact comme modèle de production.

## G0–G8 exécutés

| Gate | Résultat | Preuve |
|---|---|---|
| G0 inventaire | Pass technique | `193` latents, `8` prompts, scope `168`, digest `217fed73b66001be` |
| G1 contrat | Pass | pack/reload/tests ciblés : `12 passed`; codes, scales, scope NPZ et biais dérivé contrôlés |
| G2 référence/split | Partiel/bloquant | teacher déterministe ; préflight `--require-independent` code 2, `193/193` train-seen, aucun held-out |
| G3 sensibilité | Mesuré | erreur poids moyenne G32 `0,4591`, G64 `0,4652`, G128 `0,4699` |
| G4 sélection | Rejeté | G128 sous taille mais cosine court `0,6923`; G64 compact préférable mais qualité courte `0,7038` |
| G5 cascade | Exécuté, rejeté | `24 × 200`, puis 2 fenêtres contrôlées ; meilleur cosine `0,8346/0,7205` |
| G6 extension | Non autorisé | le core n’a pas passé G5 ; aucune extension silencieuse |
| G7 audio | Rejeté | 3 prompts ; 1 pass, 2 fails ; RMS/peak bruts conservés |
| G8 ressources | Incomplet/rejeté | fenêtre terminale : Metal peak `5,09 GiB`, RSS `~929 MB`, swap `10,05 GB` |

## Artifacts principaux

- Inventaire : `output/sample-expertise-pilot/ternary-quality-v3/g0-g2/inventory.json`
- Sensibilité : `output/sample-expertise-pilot/ternary-quality-v3/g3-sensitivity.json`
- Candidat final avant polish : `output/sample-expertise-pilot/ternary-quality-v3/final-symmetric-g64/`
- Candidat final polish sûr : `output/sample-expertise-pilot/ternary-quality-v3/final-symmetric-g64-polish-safe2/`
- Audit velocity : `final-symmetric-g64-polish-safe2/audit-debug-train-seen/`
- Audit audio brut : `final-symmetric-g64-polish-safe2/audio-gate-debug/`
- Profil ressources : `final-symmetric-g64-polish-safe2/resources.json`
- Préflight split indépendant : `g2-independent-preflight-final/` (`independent_available=false`)
- Fenêtre student-rollout rejetée : `window-rollout-g64/`
- Fenêtre terminale rejetée, meilleur snapshot : `window-terminal-g64-safe/`
- Ré-audit contrat du meilleur snapshot : `window-terminal-g64-safe/audit-contract-revalidated/`

Le fichier final compact fait `479 660 202` octets. Les biais affine ne sont
pas stockés : pour un modèle strict symétrique, ils sont reconstruits comme
`biases = -scales` au reload. Parité paramètres et forward : exactes.

## Ce qui a réellement échoué

1. **Distorsion structurelle.** La quantification core stricte garde une erreur
   de reconstruction poids autour de `0,46`. G128 économise de la taille mais
   perd davantage de capacité ; G32 améliore les poids mais dépasse la taille.
2. **Trajectoire haute sigma.** Le pire niveau est `sigma=0,95` : cosine moyen
   `0,7809` sur le candidat final. Le loss local par bloc ne garantit pas la
   fidélité du rollout complet.
3. **Polish insuffisant.** Le premier polish avait un bug AdamW : `eps=1e-8`
   sous-flotait en FP16 et créait `0/0 → NaN`. Le correctif `eps=1e-4`, clip
   `0,10` et gate de finitude supprime le NaN, mais 200 steps ne change pas la
   qualité (`0,8302 → 0,8304`).
4. **Audio non acceptable.** Le rendu piano a RMS ratio `2,676` et peak ratio
   `1,823`; le rendu rock a RMS ratio `1,371`. Aucun gain/normali­sation n’a
   masqué ces défauts.
5. **Reprise désormais instrumentée, mais non suffisante pour accepter.** Le
   checkpoint de pas persiste maintenant poids maîtres, optimizer, compteur,
   RNG Python/NumPy/MLX et filiation des données ; le test d’intégration a
   repris le même pas avec la même perte. Le snapshot historique `24 × 200`
   n’a pas été reconstruit depuis ce mécanisme, et cela ne corrige ni la
   capacité ternaire ni l’absence de held-out.

## Plan v4 / suite nécessaire après les deux fenêtres

1. **Fait techniquement.** Le checkpoint complet est implémenté et testé sur
   reprise courte ; il doit encore être utilisé pour tout futur run long.
2. **Fait comme pilote, rejeté comme correctif.** La fenêtre `20–23`, 100
   updates, LR `2e-5`, mélange états teacher/student et rollout différentiable
   2 pas a produit `0,8243/0,7123`. La fenêtre terminale `23`, LR `5e-6`, 100
   updates a produit `0,8346/0,7205`. La trajectoire d’état est pourtant
   `0,9988/0,9961` : l’écart est dans la velocity, surtout à haute sigma.
3. Tester une politique mixte **déclarée** G32/G64 par famille sensible, avec
   budget disque mesuré. Aucun déploiement G32 global s’il dépasse 500 MB.
4. Comparer strict-core, strict-core + résidu FP16 explicitement annoncé, et
   scope élargi. Le résidu est un contrôle de capacité, pas une preuve de
   modèle 100 % ternaire.
5. Fournir une banque indépendante de sources/prompts pour validation/test ;
   sceller les seeds avant choix.
6. Rejouer G7 sur 12 prompts × 3 seeds × 12/30 s, puis écoute aveugle niveau
   égal. Tant que ces éléments manquent, statut `musical_review_pending` ou
   `rejected`, jamais `accepted_local`.

## Règle d’arrêt

Ne pas relancer un run long identique. Le prochain run doit changer au moins
l’objectif de rollout, la politique de groupes ou le périmètre explicitement,
et passer le pilote haute-sigma avant toute cascade.

## Addendum expérimental — fenêtres G5 du 22 septembre 2026

Les deux fenêtres prévues ont été exécutées depuis le même snapshot source
`final-symmetric-g64-polish-safe2`, sans promotion automatique :

| Snapshot | Modification | Audit 8 prompts, 5 sigmas | Trajectoire 2 pas | Verdict |
|---|---|---:|---:|---|
| source | aucune | `0,83042 / 0,71843` | `0,99877 / 0,99606` | rejeté |
| `window-rollout-g64` | blocs 20–23, 100, LR `2e-5`, rollout student différentiable | `0,82431 / 0,71234` | `0,99878 / 0,99608` | rejeté |
| `window-terminal-g64-safe` | bloc 23, 100, LR `5e-6` | **`0,83464 / 0,72052`** | `0,99877 / 0,99614` | rejeté |

Le meilleur snapshot reste sous les gates provisoires `0,93 / 0,85`. Le rendu
brut de ce meilleur rejeté, 3 prompts × 4 s × 8 pas, donne : piano RMS `1,522`
(fail), funk `1,093` (pass), rock `1,172` (pass). Ce n’est pas une validation
musicale : l’audio a seulement servi de diagnostic technique court.

Ressources du meilleur rejeté : fichier `480 542 876` octets, RSS maximal
`929 366 016` octets, Metal après reload `0,574 GiB`, swap macOS
`10 049,69 MiB` utilisés. La contrainte « sous 12 GB » n’est donc pas démontrée
en exécution saine malgré un pic Metal modéré.

Conclusion causale : le cache et le rollout student reproduisent très bien
l’état latent, mais l’optimisation de fenêtre ne remonte pas la velocity. Le
défaut dominant reste la capacité/distorsion du core ternaire à haute sigma et
sur certains prompts, pas un simple bug de sampler. Deux fenêtres contrôlées
ayant échoué, aucune troisième cascade n’est lancée dans v3.

### Renforcement G1 post-run

Le loader d’audit valide désormais le contenu du NPZ avant de construire le
modèle MLX : dtype et longueur du packing, codes réservés, formes des scales et
des biais dérivés, signe des scales symétriques et validation du tenseur
déquantifié. Les 12 tests ciblés passent. Le meilleur snapshot a été rechargé
avec ce contrôle (`codes_and_metadata_valid=true`) ; son audit court reste
rejeté (`0,84439 / 0,80644` sur un prompt), ce qui confirme que le nouveau
contrôle de contrat ne masque pas l’échec de qualité.

## Audit des sources alternatives pour G2

Une recherche locale a trouvé deux réservoirs de latents hors du train v3 :

| Réservoir | Latents | Métadonnées droits/licence | Annotation | Décision |
|---|---:|---:|---|---|
| `output/sample-expertise-pilot/broad-music/latents-12s` | 120 | 0/120 | `provided_unreviewed` | exclu |
| `output/sample-expertise-pilot/sftminimal/latents-12s` | 58 | 0/58 | `provided_unreviewed` | exclu |

Les deux réservoirs ont `training_intent=style_and_mix_reference`, aucun champ
`role/split`, et ne figurent pas dans `configs/ternary_quality_v3.json`. Le
config SFT contient en plus la mention explicite que les droits
`training_sa3` ne sont pas autorisés. Ils ne peuvent donc pas être promus
silencieusement en validation/test indépendant. G2 reste bloquée jusqu’à
l’arrivée d’un corpus réellement autorisé et documenté.

## Continuation autorisée v4 — corpus personnel — 22 septembre 2026

L’autorisation explicite d’utiliser les corpus locaux pour étude personnelle a
levé le blocage d’usage, sans transformer leurs annotations en vérité musicale
ni autoriser une redistribution. Le builder
`services/musicgen/build_ternary_independent_corpus.py` a conservé les sources
originales, exclu les **46** sources broad déjà présentes dans le train v3 et
construit un split sans parent partagé :

| Split | Samples | Parents | Prompts |
|---|---:|---:|---:|
| train v4 | 253 | 202 | 28 |
| validation | 39 | 28 | 18 |
| test | 79 | 66 | 24 |

Préflight : `independent_available=true`, zéro chevauchement train/validation/test.
Configuration : `configs/ternary_quality_v4_authorized.json`.

### Résultat du run v4

Le run strict symétrique G64 a exécuté `24 × 200` updates, polish 200, export
compact et reload sur le SSD externe. Artefact :
`/Volumes/Extreme SSD/OnUsLoopLab/ternary-quality-v4-authorized-20260922/g64-expanded/`.

| Gate | Résultat |
|---|---|
| Contrat/reload | pass ; `168` tenseurs, parité paramètres exacte, erreur reload `0` |
| Taille | `479 660 087` octets, sous 500 MB |
| Validation held-out | rejetée : cosine moyen/pire `0,81248 / 0,65934` |
| Test held-out | rejeté : cosine moyen/pire `0,80827 / 0,69551` |
| Audio brut validation | `0/3` pass ; RMS ratios `1,610`, `1,462`, `1,484` |
| Ressources | entraînement Metal peak `5,42 GiB`; reload Metal `0,57 GiB`; RSS `1,41 GB`; swap `7,59 GB` |

Le run v4 est donc **rejeté**, malgré corpus autorisé et split indépendant. Le
résultat renforce le diagnostic : l’ajout de données ne corrige pas la
distorsion du core strict `s*q`, surtout dans les blocs tardifs (`block 23`
mean dense error `0,48034`). La prochaine expérience doit changer la capacité
ou l’objectif (student ternaire natif, périmètre hybride explicitement déclaré
ou résidu de contrôle), pas seulement ajouter des steps ou des sources.
