# Base de connaissances — DiT ternaire Stable Audio 3

**Mise à jour :** 25 septembre 2026  
**Statut :** recherche et implémentation expérimentale. Pas encore un modèle de production.  
**Objectif confirmé :** compromis Bonsai — cœur attention/FFN à codes ternaires {-1, 0, +1}, supports sensibles à leur précision native ; DiT cible 550–650 Mo, travail mesuré sous 12 000 000 000 octets Metal, qualité audio démontrée séparément.

Le [plan V9 révisé](TERNARY_QUALITY_RECOVERY_PLAN_V9.md) applique la décision
« comme Bonsai, pas plus » : 168 matrices converties, supports conservés,
pilotes G128/G32, données de trajectoire, fenêtres recouvrantes et écoute
avant généralisation. Aucun passage ultérieur au toutes-matrices n'est prévu.
Le [registre V9](TERNARY_V9_EVIDENCE_2026-09-25.md) conserve les preuves.
Aucun modèle complet n'est encore accepté. L'estimation 10–30 % est retirée
car non calibrée ; la probabilité reste inconnue, et >90 % non démontré.

Cette fiche est la mémoire projet. Elle conserve les faits vérifiés, les hypothèses, les erreurs à ne pas répéter et les gates avant toute annonce de résultat.

Les sections 3–22 sont historiques : elles conservent les résultats sans les
réécrire. En cas de divergence, le plan V9 révisé, son registre et la section 23
font référence. Le [registre initial](TERNARY_RECOVERY_EVIDENCE_2026-09-22.md)
et la [V6 archivée](archive/TERNARY_QUALITY_RECOVERY_PLAN_v6_2026-09-24.md)
restent disponibles.

## 1. Documents et code liés

Documents de recherche détaillés :

- [Investigation ternarisation](/Users/guillaumegaillard/.gemini/antigravity/brain/c040d0d7-51ff-4c74-a370-ebd12c376801/ternary_investigation.md) — source externe, peut ne pas être disponible sur une autre machine.
- [Plan d’implémentation](/Users/guillaumegaillard/.gemini/antigravity/brain/c040d0d7-51ff-4c74-a370-ebd12c376801/implementation_plan.md) — source externe, peut ne pas être disponible sur une autre machine.

Code et artefacts du dépôt :

- [train_bonsai_pure_ternary.py](../services/musicgen/train_bonsai_pure_ternary.py) — QAT H128/cascade expérimental.
- [distill_bonsai_v5_real_cascade.py](../services/musicgen/distill_bonsai_v5_real_cascade.py) — cascade group64 avec préfixe étudiant.
- [distill_bonsai_full_500mb.py](../services/musicgen/distill_bonsai_full_500mb.py) — export group64 proche de 500 MB.
- [render_bonsai_ternary_audios.py](../services/musicgen/render_bonsai_ternary_audios.py) — audit de génération et audio.
- [distill_bonsai.log](../output/sample-expertise-pilot/distill_bonsai.log) — run group64 et audit E2E.
- [artefact group64 actuel](../output/sample-expertise-pilot/universal-models/dit_medium_bonsai_ternary_int2_group64.npz) — référence mesurée, environ 493,8 MB.

Code de la voie de récupération :

- [ternary_contract.py](../services/musicgen/ternary_contract.py) — contrat q/scales/biases et packing exact.
- [train_ternary_quality.py](../services/musicgen/train_ternary_quality.py) — cascade group64/group32 avec reload gate.
- [audit_ternary_quality.py](../services/musicgen/audit_ternary_quality.py) — velocity audit sur artefact rechargé.
- [render_ternary_quality.py](../services/musicgen/render_ternary_quality.py) — audio brut sans normalisation.
- [finetune_ternary_window.py](../services/musicgen/finetune_ternary_window.py) — E2E fenêtre sous contrainte VRAM.
- [ternary_adapter.py](../services/musicgen/ternary_adapter.py) et [train_ternary_adapters.py](../services/musicgen/train_ternary_adapters.py) — prototype d’adapter low-rank séparé, non validé audio.

## 2. État historique avant exécution v3

Ni la fidélité QAT/export/reload, ni la conformité mémoire système, ni la qualité
musicale ne sont clôturées. Les fichiers se chargent, mais des casts de buffers,
des mesures manquantes et l’absence de test indépendant empêchent une conclusion
plus forte. Priorité d'alors : P0 du [plan v3 archivé](archive/TERNARY_QUALITY_RECOVERY_PLAN_v3_2026-09-22.md), sans
relancer immédiatement l’entraînement.

### 2.1 Résultat vérifié du run complet

Artefacts produits sous `output/sample-expertise-pilot/ternary-quality-recovery/` :

| Artefact | Résultat |
|---|---|
| `group64-core` | 512,63 MB ; velocity 0,8076/0,6780 ; audio spectral 0,1618 |
| `group32-core` | 580,27 MB ; velocity 0,8475/0,7130 ; audio spectral 0,2408 |
| `group32-rollout20-23` | velocity 0,8127/0,6924 ; audio spectral 0,2279 |
| `group32-adapter-r8` | +34,02 MB ; audio spectral 0,1548 ; velocity 0,8442/0,7367 rapportée précédemment sans JSON dédié retrouvé |
| `group32-adapter-real` | +34,01 MB ; velocity 0,8682/0,7476 ; audio spectral 0,1515 |

Le cosinus reload ≥0,999997 masque des erreurs relatives maximales de 0,2153 %
et 0,2489 %, supérieures à l’objectif initial de 0,1 %. La métrique
`max_dense_reconstruction_error` est renseignée à zéro sans calcul ; elle n’est
pas probante. L’export convertit notamment les fréquences temporelles FP32 en
FP16 : la sonde du registre confirme une modification importante des features
sin/cos, dont l’effet audio reste à isoler.

Les valeurs mémoire sont des GiB de l’allocateur Metal : 29,42 pour la QAT
globale testée, 7,78 pour une fenêtre de quatre blocs, environ 5,95 pour un
adapter. RSS/swap incomplets ; ni conformité système ni impossibilité générale
de QAT sous 12 GB ne sont démontrées.

Les audits réutilisent huit prompts/exemples d’entraînement ; les mesures audio
principales portent sur un seul piano et le canal gauche. Les adapters sont
FP32, pas FP16. L’effet du groupe n’est pas isolé entre les runs 64 et 32.
Les états student et les seuils appris restent des hypothèses à tester après
correction du contrat, pas des remèdes garantis.

## 3. Architecture à garder en tête

Le DiT contient 24 blocs. Les projections principales sont :

- self-attention : to_qkv, to_out ;
- cross-attention : to_q, to_kv, to_out ;
- FFN : ff.0.proj, ff.2 ;
- conditionnement local : to_local_embed.seq.0 avec entrée 257, puis seq.2 avec entrée 1536 ;
- paramètres continus : normes, to_scale_shift_gate, project_in/out et embedder global.

Conséquence : « toutes les projections attention/FFN sont ternaires » ne signifie pas « tout le DiT est ternaire ».

### Périmètres à nommer séparément

| Livrable | Scope | Ce qui reste FP16 |
|---|---|---|
| core-ternary | 7 projections par bloc, 168 matrices | local conditioning, entrées/sorties, embedder, normes, gates |
| group64 actuel | 8 matrices compatibles par bloc, 192 matrices | seq.0 et autres poids hors prédicat |
| DiT-matrices-ternary, cible v3 | toutes les matrices apprises du DiT couvertes | biais vectoriels, normes/gates, scales et buffers : précision et coût explicités |

Ne pas utiliser « 100 % ternaire » sans dénominateur. Même le troisième
périmètre ne ternarise pas nécessairement tous les scalaires. Le text encoder
et le codec sont également hors DiT.

Le quantizer actuel produit `m+s*q`, donc trois niveaux affines dont le centre
n’est pas forcément zéro. La cible stricte v3 est `s*q` ; Hadamard, si retenu,
donne des codes ternaires en base tournée, pas des poids ternaires dans la base
d’origine. Toute LoRA non fusionnée rend la variante hybride.

## 4. Preuves locales disponibles

### 4.1 Artefact group64

Le run [distill_bonsai.log](../output/sample-expertise-pilot/distill_bonsai.log) montre :

- 193 pistes chargées ;
- 24 blocs traités ;
- 192 couches linéaires quantifiées ;
- CosSim local des blocs environ 0,9900–0,9988 ;
- export de 493,8 MB ;
- audit E2E sur 24 lignes : moyenne 0,5596, minimum 0,1735, maximum 0,8123 ;
- moyennes par prompt : piano 0,5866, funk 0,5436, ambient 0,5486.

Le signal important est le suivant : un bon CosSim de bloc ne garantit pas une bonne trajectoire de diffusion ni un bon audio.

### 4.2 Bloc 0 H128

Valeurs déclarées dans les notes historiques :

| Variante | CosSim | Ratio norme | MSE | Confiance |
|---|---:|---:|---:|---|
| ternarisation brute | 0,9292 | 0,7163 | 5,0178 | à reproduire |
| H128 | 0,9527 | 0,8505 | 2,8340 | à reproduire |

Le gain MSE annoncé de 43,5 % est une observation de couche à confirmer. Il ne prouve pas un gain E2E.

À sauvegarder lors de la reproduction :

- seed, dtype, sigma/timestep, crop et positions audio/mémoire ;
- convention de H128 et contrôle H.T @ H ≈ I ;
- sorties teacher, QAT, export et reload ;
- CosSim, MSE, norme, biais DC et histogramme {-1,0,+1} ;
- temps réel par couche et par étape de diffusion.

### 4.3 Valeurs historiques non vérifiées

Les valeurs suivantes ne sont pas des gates passés tant qu’un log reproductible n’est pas archivé :

- CosSim latent historique 0,0473 ;
- peak audio 0,010 contre 1,46 ;
- pic VRAM 19,19 GB ;
- cascade estimée 2,4 GB ;
- polish estimé 4,5 GB ;
- artefact H128 annoncé à 455,8 MB.

## 5. Erreurs connues

### 5.1 Mismatch H128 entre QAT et export

Le QAT H128 fait une rotation puis une rotation inverse dans le forward. L’export actuel peut re-quantifier des groupes bruts sans la même convention. Le modèle peut donc bien s’entraîner puis changer de fonction au reload.

Règle : une seule fonction de quantification doit servir au QAT, au repacking, au reload et à l’inférence.

### 5.2 Liste de couches incohérente

Le QAT du script H128 remplace 7 couches par bloc. L’export générique peut inclure seq.2, tandis que le renderer ne traite que les chemins attention/FFN. Une couche entraînée, exportée et chargée avec trois scopes différents invalide l’audit.

Règle : écrire le scope exact dans le manifeste et faire échouer le run si les trois listes diffèrent.

### 5.3 Student forcing incorrect

Entraîner chaque bloc sur les sorties teacher puis l’exécuter avec les sorties étudiantes crée un biais d’exposition. La cascade finale doit utiliser le préfixe étudiant réellement quantifié et figé.

### 5.4 Normes et AdaLN

La dérive de norme peut transformer une erreur de poids modérée en quasi-silence après plusieurs blocs et étapes. La première version garde normes, scales, shifts et gates en FP16 ; leur calibration doit être séparée et mesurée.

### 5.5 Score trop local

Les tests de conformité des codes ternaires et les CosSim de blocs sont nécessaires, mais insuffisants. Le gate utile est : même bruit, même prompt, même sigma, trajectoire E2E, puis audio décodé.

## 6. Contrat de quantification v3

Un snapshot unique de codes/scales à la précision déployée alimente QAT gelée,
packing, reload et inférence. Pas de re-quantification silencieuse à l’export,
pas de cast global des buffers, pas de métrique synthétique égale à zéro.

Première voie v3 : `W_hat=s*q`, sans centrage ni rotation, groupe 64 ; groupe
128, seuil appris ou rotation sont des challengers contrôlés. Conserver poids
maîtres, quantizer, optimiseur et RNG dans un checkpoint distinct du paquet
d’inférence. Déquantifier les centres ternaires ne restaure pas cet état.

Hadamard reste optionnel : pour `W_r=W R`, `W_hat=Q R.T`, le forward utilise
`(x R) Q.T`. Tester convention, normalisation, padding et kernel. Taille de
rotation et taille de groupe sont deux paramètres différents. Détails et
tolérances : section 4 et gate G1 du plan v3.

## 7. Cascade de distillation v3

Pour chaque bloc i de 0 à 23 :

1. charger les cibles teacher ;
2. exécuter le préfixe étudiant déjà ternarisé ;
3. entraîner seulement le bloc i ;
4. utiliser plusieurs sigma/timesteps, crops, prompts et genres ;
5. comparer hidden state, velocity, RMS, DC et relation temporelle ;
6. figer le bloc après test save/reload ;
7. libérer les activations avant le bloc suivant.

Le protocole v3 ajoute des états réellement rencontrés par le student et
interroge le teacher sur ces mêmes entrées. Un cache de trajectoires teacher
n’est ni du student forcing ni une loss de rollout multi-pas.

Objectif initial : NMSE de velocity et faible terme de direction ; pertes
supplémentaires seulement par ablation. Les fenêtres optimisent la sortie du
DiT complet, avec gradient à travers le suffixe gelé. Allowlist explicite des
paramètres ; scales entraînables seulement avec opérateur compatible. Les
budgets, checkpoints et critères d’arrêt sont fixés dans le plan v3.

## 8. Budget mémoire

Mesurer séparément :

- chargement teacher ;
- préparation des latents ;
- QAT d’un bloc ;
- cascade complète ;
- polish ;
- export/reload ;
- génération et décodage audio.

Journal minimal : pic Metal actif/réservé, RSS, swap, batch, longueur latente, dtype et objets teacher/student présents.

Garde v3 : 11 000 000 000 octets ; limite 12 000 000 000 octets. Les anciens
compteurs `_gb` utilisent en réalité `1024**3` : ce sont des GiB. Ne pas
additionner RSS et Metal sans traiter leur recouvrement en mémoire unifiée.
Les estimations ne deviennent des résultats qu’après instrumentation.

## 9. Taille disque

455,8 MB est une cible de comptabilité, pas un artefact existant.

Chaque export doit publier :

- nombre de poids et de groupes par tenseur ;
- bytes de codes packés ;
- bytes de scales, biais et métadonnées ;
- poids laissés FP16 et raison ;
- taille théorique ;
- taille réelle sur disque ;
- résultat du reload.

Le format 2-bit stocké n’est pas automatiquement 1,58 bit effectif : scales, padding, conteneur et métadonnées comptent.

Relevé v3 : group64 = 512 629 358 octets sur disque, 613 104 896 octets de
tenseurs ; group32 = 580 269 830 et 698 039 552. Les coûts détaillés et la
distinction compression ZIP/mémoire des poids figurent dans le registre.

## 10. Gates avant annonce

Numérotation canonique : plan v3, G0 inventaire, G1 contrat, G2 référence et
splits, G3 sensibilité, G4 sélection courte, G5 distillation, G6 périmètre et
taille, G7 trajectoires/audio/écoute, G8 ressources et paquet.

Codes et buffers sérialisés doivent être préservés exactement. Les tolérances
kernel sont calibrées par dtype avant entraînement. Les seuils velocity
proposés filtrent les candidats, sans certifier leur qualité musicale.

Le seuil spectral universel 0,85 est retiré. Mesurer stéréo, durée réelle,
bruts avant gain, diversité et adéquation ; calibrer les proxies contre des
contrôles et une écoute indépendante. Un nouveau seed sur les mêmes données
d’entraînement n’est pas un test held-out.

## 11. Recherche de référence

- [TerDiT](https://arxiv.org/abs/2405.14854) : QAT ternaire pour DiT et traitement spécifique de l’AdaLN.
- [Post-Training Quantization for Audio Diffusion Transformers](https://arxiv.org/abs/2510.00313) : Stable Audio Open, calibration dépendante du timestep et compensation résiduelle.
- [TQ-DiT](https://arxiv.org/abs/2502.04056), [LRQ-DiT](https://arxiv.org/abs/2508.03485), [HadaNorm](https://arxiv.org/abs/2506.09932) : regroupement temporel, rotation adaptative, centrage et outliers.
- [Q-VDiT](https://proceedings.mlr.press/v267/feng25q.html), [MPQ-DMv2](https://arxiv.org/abs/2507.04290) : distillation des relations temporelles et quantification mixte.
- [BitNet b1.58](https://arxiv.org/abs/2504.12285), [BitDistill](https://arxiv.org/abs/2510.13998) : poids maîtres haute précision, warm-up et distillation multi-signal.
- [CAT-Q, juin 2026](https://arxiv.org/abs/2606.26650) : modulation et ternarisation douce pour LLM ; piste d’optimisation, pas preuve audio.
- [Ternary Bonsai](https://prismml.com/news/ternary-bonsai), [Bonsai Image 4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-unpacked), [Bonsai 2 demo](https://github.com/PrismML-Eng/Bonsai-demo) : preuves d’ingénierie et nécessité d’un runtime/kernels cohérent avec H.

Ces travaux sont des références de méthode. Aucun ne valide à lui seul un DiT audio Stable Audio 3 entièrement ternaire.

La revue ciblée du plan v3 distingue dates, domaines et limites de transfert.
Les références historiques supplémentaires de cette fiche ne constituent pas
une reproduction ni une validation du pipeline local.

## 12. Procédure de reprise v3

1. corriger les mesures, dtypes/buffers et l’équivalence des forwards ;
2. refaire l’audit logiciel des artefacts existants, sans nouvelle QAT ;
3. établir référence, splits indépendants et budget mémoire/taille ;
4. mesurer la sensibilité et choisir un seul contrat par petits essais ;
5. distiller sur états réels/teacher/student, avec reprise fidèle ;
6. évaluer le fichier rechargé sur trajectoires, stéréo et écoute ;
7. étendre le périmètre et compresser sans perdre les validations ;
8. publier résultats et limites, jamais une réussite fondée sur le seul export.

**Règle de mémoire :** si un résultat n’a pas son seed, sa config, son artefact, son log et sa métrique de reload, le conserver comme hypothèse, jamais comme fait.

## 13. Exécution v3 du 22 septembre 2026

Le rapport complet est [TERNARY_V3_EXECUTION_REPORT_2026-09-22.md](TERNARY_V3_EXECUTION_REPORT_2026-09-22.md).

Résultat : le contrat strict symétrique, le compactage mono-scale, le reload et
la cascade `24 × 200` fonctionnent. Le candidat G64 compact fait `479 660 202`
octets, mais reste rejeté : velocity `0,8304` moyen / `0,7184` pire sur
`debug_train_seen`, puis 2/3 rendus audio hors gate brut. G128 passe aussi la
taille mais reste à `0,6923` sur le pilote ; G32 améliore la reconstruction
poids mais dépasse `500 000 000` octets et ne passe pas le pilote.

Le polish FP16 a d’abord exposé un bug général : AdamW `eps=1e-8` sous-flotte
en FP16. `eps=1e-4`, clip et contrôle de finitude corrigent le NaN, sans
amélioration qualité mesurable. La conclusion actuelle est donc **rejeté,
expérimental**, pas « modèle ternaire de qualité ».

## 14. Fenêtres G5 réellement exécutées — 22 septembre 2026

Deux fenêtres ont été lancées depuis le même snapshot G64 compact. Elles sont
des expériences rejetées, pas des checkpoints de production.

| Run | Périmètre | Résultat velocity (moy./pire) | État latent rollout | Audio brut court |
|---|---|---:|---:|---:|
| source | aucun retrain | `0,83042 / 0,71843` | `0,99877 / 0,99606` | référence |
| `window-rollout-g64` | blocs 20–23, 100 updates, LR `2e-5`, 2 pas student différentiables | `0,82431 / 0,71234` | `0,99878 / 0,99608` | non promu |
| `window-terminal-g64-safe` | bloc 23, 100 updates, LR `5e-6` | **`0,83464 / 0,72052`** | `0,99877 / 0,99614` | 2/3 technique pass |

Le second run est le meilleur des deux, mais reste très inférieur aux gates
provisoires `0,93 / 0,85`. Son test audio brut 3 prompts × 4 s × 8 pas donne
RMS piano `1,522` (rejet), funk `1,093` et rock `1,172` (technique pass). Ne
pas confondre état latent fidèle et velocity fidèle : ici l’état de trajectoire
est quasi identique, tandis que la sortie teacher/student reste très éloignée.

Le script de fenêtre persiste désormais le cache teacher/student, les bruits
de re-noise, les sigmas, et peut rétropropager une perte sur deux pas student.
Le pilote montre que cette amélioration de protocole ne suffit pas à compenser
la distorsion du core ternaire à haute sigma. Après deux fenêtres contrôlées,
la règle d’arrêt est appliquée : pas de troisième cascade sous le même contrat.

Ressources du meilleur rejeté : artefact `480 542 876` octets, RSS maximal
`929 366 016` octets, Metal après reload `0,574 GiB`, swap macOS
`10 049,69 MiB`. Le budget Metal seul ne constitue pas une preuve « sous
12 GB » sur cette machine.

### G2 — sources alternatives inspectées

La recherche locale a aussi inspecté `broad-music/latents-12s` (120 latents)
et `sftminimal/latents-12s` (58). Les deux réservoirs ont zéro champ
droit/licence, `source_annotation_status=provided_unreviewed`, aucun rôle de
split et ne sont pas déclarés dans la configuration v3. Le dataset SFT contient
en outre un avertissement explicite sur les droits `training_sa3`. Ils restent
hors validation/test : un fichier présent dans le workspace n’est pas, à lui
seul, une autorisation d’usage.

## 15. G1 renforcé après le run

Un manifest correct ne suffit pas : le loader doit aussi inspecter les arrays
du NPZ avant de construire le modèle. Le contrôle ajouté vérifie le dtype et la
longueur du packing, l’absence de codes réservés, les formes `scales`/`biases`,
la dérivation `biases = -scales` en compact symétrique, les signes et la
validation du tenseur déquantifié. Les 12 tests ternaires ciblés passent.

Le meilleur snapshot `window-terminal-g64-safe` passe ce contrôle (`168`
tenseurs, codes/scales valides), mais son audit court reste rejeté à
`0,84439 / 0,80644` (moyenne/pire). La validation du contrat est donc une
condition nécessaire, jamais un substitut à la validation de qualité.

## 16. Corpus autorisé et run v4

L’autorisation explicite d’utiliser les corpus locaux pour étude personnelle a
permis d’ouvrir G2, tout en conservant la provenance et l’interdiction de
redistribution. Le builder a comparé les sources par parent : 46 sources broad
étaient déjà dans le train v3 et ont été exclues du held-out ; 132 sources
restaient éligibles.

Split v4 : train `253` samples / `202` parents / `28` prompts ; validation `39`
samples / `28` parents / `18` prompts ; test `79` samples / `66` parents / `24`
prompts. Aucun identifiant de parent selon les noms ne traverse les splits ;
l'absence de doublons audio ou de lignées communes reste à vérifier.

Le run G64 symétrique `24 × 200` + polish a exporté `479 660 087` octets et
rechargé avec parité exacte. Résultats des jeux désignés held-out, sans preuve
complète d'indépendance : validation
`0,81248 / 0,65934`, test `0,80827 / 0,69551` (moyenne/pire). Audio brut de
trois prompts validation : `0/3` pass, RMS ratios `1,610 / 1,462 / 1,484`.

Conclusion révisée : cette expérience n'atteint pas les critères employés.
Elle n'isole pas un effet de volume de données et ne démontre pas une limite
intrinsèque du ternaire. Garder les jeux v4 comme développement historique,
pas comme nouveau test scellé. Voir corrections ci-dessous.

## 17. Révision du diagnostic et plan v6 — 23 septembre 2026

Aucun nouveau run d'entraînement pendant cette révision. Lecture du code,
rapports, en-têtes du checkpoint et sources de recherche uniquement.

### Limites à ne plus transformer en certitudes

- `parent_key()` dans `prepare_ternary_dataset.py` hache un chemin ou nom.
  « Aucun parent commun » signifie seulement aucun identifiant commun selon
  cette règle. Il faut encore déduplication PCM, lignées et exposition passée.
- `heldout=true` dans l'audit découle de `--split-role`, pas d'une vérification
  de l'indépendance. Les anciens tests consultés sont désormais du développement.
- La trace de `audit_ternary_quality.py` omet le latent après dernier pas.
  Cosinus élevés des états bruités ne prouvent pas une trajectoire finale fidèle.
- Le trainer v4 n'applique pas l'accumulation 4 ni les LR/weight decay annoncés
  dans le JSON de plan. Sa configuration d'exécution est la preuve utile.
- L'erreur de reconstruction des poids, environ 0,48 au bloc 23, ne prouve pas
  une insuffisance irréductible de capacité. La QAT locale irréversible,
  l'exposition limitée et les distributions de timesteps restent des hypothèses.
- Le swap système absolu ne mesure pas le swap induit par le run. Mesurer
  baseline, delta et pression mémoire ; ne pas conclure sur Metal seul.

### Levier de taille vérifié par inventaire

Le checkpoint `dit_medium_f16.npz` contient 525 tenseurs,
1 453 368 336 éléments et 2 907 132 960 octets de tenseurs.
Les 230 tenseurs de dimension ≥2 contiennent 1 452 609 536 éléments ; ce
nombre inclut le conditionneur de durée, distinct du DiT proprement dit.

Lecture avec `zipfile` et les lecteurs publics d'en-tête NPY de NumPy :
pour chaque matrice/Conv, aplatir les dimensions après l'axe de sortie pour le
comptage, arrondir chaque ligne au groupe G, puis compter N_padded/4 octets de
codes, 2*N_padded/G octets de scales et les autres tenseurs à leur dtype.
Ce calcul ne valide pas encore la disposition Conv ni son kernel.

Résultats calculés, avant conteneur/buffers additionnels/rotation :

- toutes matrices G32 : **455 818 272 octets** ;
- toutes matrices G64 : **410 720 288 octets** ;
- toutes matrices G128 : **388 613 664 octets**.

Un G32 intégral peut donc respecter la cible, même si un core-G32 avec le
reste dense la dépasse. Ce n'est ni un export livré ni une preuve qualité.
Le coût mémoire MLX peut inclure des biais dérivés supplémentaires par groupe.

### Décision méthodologique et recherche

Le plan actif garde le préentraînement. Calibration apprise, transition
douce/dure, validation toujours dure, puis QAT par fenêtres réouvertes et
objectif velocity global. Pas de nouveau modèle depuis zéro par défaut.
Le pilotage porte sur erreur de sortie, couverture et écoute, pas uniquement
sur proximité des poids ou taille d'un NPZ.

La revue actualisée ajoute notamment
[RobuQ, révision mai 2026](https://arxiv.org/html/2509.23582v2), qui évalue
des DiT image préentraînés avec un budget bien supérieur aux essais locaux,
et distingue les exceptions haute précision. Les détails, limites de transfert
et liens CAT-Q, ParetoQ, audio PTQ et Bonsai sont dans la section recherche du
[plan v6 archivé](archive/TERNARY_QUALITY_RECOVERY_PLAN_v6_2026-09-24.md).

Validation audio : séparer conformité technique et acceptation musicale.
Garder le brut et les sources, comparer à niveau égal,
documenter l'autorisation locale sans prétendre à des droits de redistribution.

Prochaine étape historique : contrat/config/audit fiables, cache limité, pilote apparié.
Le disque a été libéré depuis ; vérifier les volumes avant chaque entraînement
et conserver une marge pour les checkpoints.

## 18. Exécution v6 et correctifs confirmés — 23 septembre 2026

Compte rendu historique. Les règles de reprise depuis records et la priorité
à deux pas tardifs sont révisées par les preuves de la section 19 ; ne pas
les utiliser comme procédure active.

### Défaut critique : recomputation sans gradient des paramètres du module

`train_ternary_window_v6.py` appelait `mx.checkpoint` sur une fonction qui
capturait le module et ses paramètres dans une closure. Le modèle pouvait
produire une loss, mais poids et seuils ne recevaient pas de gradient utile.
Cela explique au moins une part importante des anciens résultats qui semblaient
entraîner sans améliorer le modèle. Ne pas interpréter ces runs comme du QAT.

Le correctif rend le PyTree de `module.trainable_parameters()` explicite dans
les arguments de la fonction checkpointée, puis met à jour le module depuis
ce PyTree. Une régression démontre des gradients non nuls pour poids et seuil ;
une mise à jour corrigée a mesuré norme de gradient du seuil 0,01809 et delta
moyen de paramètres 0,000797. C'est le pattern suivi par
[l'entraîneur mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py).

### Parité d'affectation et seuil appris

Le hard-freeze et l'export n'utilisaient pas jusque-là le même seuil
d'affectation : le forward utilisait le seuil appris et l'export la scale de
reconstruction. Le contrat commun réalise désormais l'affectation apprise
bornée pour les deux chemins ; un test de parité couvre les matrices visées.
La variante à seuil appris G32 n'a pas battu le contrôle symétrique fixe au
milieu (velocity moyenne/minimum 0,97362/0,85381 contre 0,97419/0,86541).
Choix de travail : seuil fixe symétrique G32, avec scales de reconstruction
apprises ; garder l'apprentissage du seuil comme ablation, pas comme défaut.

### Mémoire, cache enseignant, état des tests

Le run QAT avec teacher en ligne a approché 11,65 GiB et a été arrêté sans
checkpoint partiel. Les cibles teacher fp16 pré-calculées (512 états,
`targets.npz`, 197 s) ont réduit le pic à 6,02 GiB sur le pilote fixe, contre
8,94 GiB pour l'ablation à seuil appris. La comparaison première mise à jour
cache/online diffère d'environ 1,5e-6 en loss. Valider le fingerprint des
poids teacher, états et cache avant chaque reprise.

Seize tests ciblés ont passé avant la cascade : contrat, parité, reload,
provenance des cibles et gradients du checkpointing. Compilation Python
passée également. Ces contrôles prouvent la cohérence logicielle couverte,
pas la qualité musicale ni l'exhaustivité de l'export.

### Résultats des pilotes correctifs

Les trois fenêtres G32 symétriques fixes 0–1, 11–12 et 22–23 ont chacune reçu
250 mises à jour. Les audits ont utilisé 18 prompts, 5 sigmas et 8 pas du split
de développement déjà consulté : ce n'est pas un test indépendant scellé.

| Fenêtre | Cosinus velocity moyen / pire | Erreur relative moyenne | État final moyen | Verdict gate |
| --- | ---: | ---: | ---: | --- |
| Tête 0–1 | 0,96585 / 0,71609 | 0,24626 | 0,90053 | échec |
| Milieu 11–12 | 0,97419 / 0,86541 | 0,20642 | 0,87300 | passe |
| Queue 22–23 | 0,95823 / 0,90507 | 0,28262 | 0,95768 | passe |

À la tête, le mauvais point sigma 0,95 a un RMS ratio 1,0499 et un cosine
0,71609 ; l'erreur est surtout directionnelle/conditionnelle, pas un collapse
de magnitude. En conséquence, la cascade reste exploratoire tant que le cumul
des fenêtres n'a pas été audité. La fenêtre 11–12 fixe bat légèrement
l'ablation seuil appris ; celle-ci modifiait environ 0,68 % des codes.

### Règle de poursuite

Poursuivre par paires séquentielles en reprenant les records du checkpoint
cumulatif précédent : 2–3, 4–5, …, 20–21, puis 22–23. Auditer et vérifier le
rechargement à chaque jalon ; interrompre ou corriger si le pire cas régresse.
La cible de taille théorique de 455,8 MB pour toutes les matrices G32 n'est
ni un export achevé ni une qualité démontrée. Le core est toujours incomplet ;
reste à mesurer seconde seed, G64 apparié, durée réelle, fidélité audio brute,
écoute A/B et budget global du paquet.

### Alerte cascade séquentielle et réouverture des fenêtres

L'audit cumulatif G32 fixe des blocs 0–3 donne 0,95104/0,72228 en cosine
velocity moyen/minimum, contre 0,96585/0,71609 pour 0–1 seul ; le terminal
moyen baisse de 0,90053 à 0,88324. Les moyennes baissent à chacun des cinq
sigmas, de 0,015 à 0,027. Le pire cas reste le même prompt (rekorder-3.2,
sigma 0,95), avec RMS ratio 1,04593 : pas de collapse d'amplitude. Le train
loss de 2–3 baisse à 0,07054 mais l'audit de développement se dégrade.
Décision : ne pas poursuivre 4–5 avec le préfixe gelé.

Le précédent trainer savait appliquer des records source mais ne pouvait pas
les rendre entraînables : la couche MLX quantifiée n'est pas `nn.Linear`. Le
nouveau helper `quantized_linear_to_qat` reconstruit le poids maître depuis
les codes/scales/biais sérialisés et conserve le biais de la couche. Tests
forward avec et sans biais, plus smoke réel des 28 modules 0–3 : réouverture
fonctionnelle, loss 0,10620 à l'unique update, pic Metal 7,45 GiB. Le forward
avant/après réouverture est égal dans la tolérance de réduction mesurée.

La passe corrective conjointe 0–3 a bien été réalisée (250 updates, cache
teacher). Elle améliore très peu le séquentiel : velocity cosine
0,95216/0,72978, erreur relative 0,29407, terminal moyen 0,88582. Gain contre
le cumul séquentiel : +0,00112 moyen, +0,00750 minimum, +0,00258 terminal ;
le gate candidat reste manqué (≥0,90/0,80). Pire cas inchangé : rekorder-3.2,
sigma 0,95, RMS ratio 1,04189. Train loss final 0,06766 ; pic 8,71 GiB.
Artefact reloadable 2 324 603 392 octets, scope 28 exact. Dix-huit tests
ciblés et compilation Python passent après l'ajout de la réouverture.

L'hypothèse « cache 512 trop étroite en états/noises » a été testée avec
2 048 états puis un corpus source-disjoint. Ces essais donnent un petit gain
ponctuel, mais pas la fidélité terminale requise ; le détail de l'essai
on-policy et la décision actualisée figurent dans la section datée ci-dessous.

### 23/09/2026 — validation disjointe, états student et échec de rollout

Le split SFT Voices préparé pour v6 est disjoint par source/parent, prompt et
hash latent : train 434 latents, 383 parents, 85 prompts ; validation 33
latents/parents et 12 prompts ; test 41 latents/parents et 24 prompts. Le
train contient 253 exemples initiaux et 181 ajouts SFT (57 prompts). L'audit
utilise les 12 prompts de validation, seed 4242, cinq sigmas et un rollout
stochastique de 8 pas. Les 24 prompts de test restent scellés.

L'auditeur attendait un dictionnaire roles.validation mais le manifeste
source-disjoint publie validation.sources[].staged_latent. La fonction
load_split_manifest accepte maintenant les deux structures et continue à
vérifier que chaque chemin sélectionné appartient au dataset fourni. Test de
régression ajouté ; les 21 tests ciblés passent et les outils modifiés
compilent.

| Audit, même validation 12 prompts | Vitesse ponctuelle mean/min | Gate ponctuel candidat/release | État terminal mean/min |
| --- | ---: | --- | ---: |
| Ancien checkpoint cache 2048 (contrôle) | 0,93584 / 0,80593 | passe / échoue | 0,61383 / 0,45651 |
| Corpus SFT, états teacher | 0,94239 / 0,84282 | passe / échoue | 0,64923 / 0,51250 |
| Student trajectories, cache 2048, 8 pas | 0,94431 / 0,79815 | échoue / échoue | 0,66628 / 0,57049 |
| Student trajectories, cache 2720, 16 pas, sigma pondérés | 0,94813 / 0,87302 | passe / passe | 0,66579 / 0,54982 |

Tous les raffinements ouvrent les mêmes 28 projections des blocs 0–3, G32
symétrique, 250 updates, à partir des records du précédent student. Les
cibles teacher sont fp16 et vérifiées par empreinte. La cache 2048 on-policy
contient 1 024 états bruités et 1 024 états student ; le lot ciblé suivant
contient 1 360 de chaque type et 16 états par rollout pour couvrir les
85 prompts de train.

Le dernier rafraîchissement fait passer le gate ponctuel de release, mais
n'améliore pas le terminal face au cache on-policy précédent : 0,66579 contre
0,66628 mean, pire 0,54982 contre 0,57049. Les états teacher/student restent
presque identiques au début puis divergent tard : mean state cosine 0,9982
(σ=0,891), 0,9792 (σ=0,746), 0,8779 (σ=0,512), 0,7306 (σ=0,274), terminal
mean 0,6658. Le pire terminal est le prompt « voice, a-cappella; 136 BPM »,
cosine 0,5498, ratio RMS 0,8802. Conclusion de diagnostic : exposition
student et sigma reweighting améliorent la proximité ponctuelle, mais ne
minimisent pas assez l'erreur intégrée du sampler. Ce n'est pas une validation
de qualité audio.

Le dernier QAT a loss first/last 0,07841/0,10745 (bruitée, pas d'amélioration
finale), pic Metal 8,74 Gio. Le cache teacher [2720,256,128] culmine à
3,69 Gio ; export cumulatif 2 324 603 402 octets, scope exact, reload exact.
Il reste 20 blocs denses : les 455,8 Mo annoncés pour le modèle entièrement
ternaire ne sont pas atteints.

Décision : ne pas avancer à 4–5, ne pas ouvrir le test. Prochaine expérience
distincte, si poursuivie : une loss différentiable de rollout sur deux pas
tardifs (σ≈0,512 puis 0,274), comparer la sortie terminale du student et du
teacher avec le même état initial, prompt et bruit sur train uniquement.
Garder le gate ponctuel et le terminal comme critères séparés ; après cette
série adaptative, réserver un jeu final neuf avant de parler d'acceptation.

## 19. Audit et plan V7 — 24 septembre 2026

Référence détaillée : [rapport forensique](TERNARY_V7_FORENSIC_REPORT_2026-09-24.md).
Plan actif : [V7](TERNARY_QUALITY_RECOVERY_PLAN.md). Cette section corrige
l'interprétation de V6 sans effacer ses résultats.

### Trois preuves nouvelles

1. **Codes inchangés entre snapshots.** Cinq raffinements de 250 updates des
   mêmes 28 matrices conservent les 226 492 416 codes, bit pour bit ; environ
   86 % des scales diffèrent à chaque transition. Cela n'exclut pas des
   flips intermédiaires annulés. Les reprises reconstruisent les maîtres
   depuis les poids déquantifiés et recréent Adam ; elles ne prolongent pas
   l'état d'optimisation antérieur. Un forward identique à la réouverture
   ne prouve pas une reprise d'entraînement fidèle.
2. **Timesteps incohérents.** La génération transmet t FP32, les cibles et
   le trainer V6 t FP16. Sur 24 états TRAIN appariés, ce seul arrondi produit
   jusqu'à 9,55 % d'erreur relative teacher. La correction passée des
   fréquences Fourier ne suffisait donc pas. Ce défaut n'est pas la cause
   unique : le student échoue aussi à sigma=1, valeur représentée exactement.
3. **Biais FFN abandonnés par records-only.** Huit biais vectoriels sont
   entraînables sur 0–3, soit 55 296 scalaires. Le format conserve le champ
   biases du quantizer, pas le paramètre bias de la couche. Un contrôle
   +0,125 est perdu au roundtrip ; les huit biais de l'export réel sont
   ceux du teacher. L'amplitude des mises à jour historiques est inconnue.

Le reload exact V6 concernait le modèle rematérialisé. Il ne clôturait pas
la parité dernier modèle entraîné → artefact. Les tests antérieurs restent
valables pour les chemins qu'ils couvraient, mais leur couverture était
insuffisante pour cette conclusion.

### Ce qu'il ne faut plus conclure

- La divergence visible tard ne démontre pas que les premières étapes sont
  bonnes : la réinjection de bruit commun masque des erreurs de débruitage.
  La prochaine expérience ne commence pas par deux pas tardifs uniquement.
- Le gain faible de cinq passes ne prouve pas qu'un apprentissage des codes
  sur ce corpus est impossible ; les reprises ont conservé les mêmes codes
  finaux et perdu l'état maître.
- Le mode symmetric retenu n'apprend pas des scales indépendantes. Le mode
  learned_symmetric essayé n'implémente pas un calendrier soft-to-hard :
  surrogate à sharpness fixe. Ne pas le nommer reproduction de CAT-Q.
- RobuQ utilise aussi une branche résiduelle FP de bas rang ; Bonsai Image
  garde des tenseurs de support FP16. Leur réussite ne prouve pas notre
  périmètre strict toutes matrices. Sources primaires et limites actualisées
  dans la section recherche du plan V7.
- « Plus de corpus », « plus de steps » ou « exporter sous 500 Mo » ne sont
  pas des corrections des trois défauts logiciels.

### Audio ajouté et statut exact

Trois paires V6/teacher ont été générées sur la validation déjà consultée :
huit pas, seeds 4242–4244, 128 latents, condition 12 s, SAME-L. Six WAV
bruts stéréo 44,1 kHz, durée réelle 11,8886 s ; aucune normalisation.

RMS ratios : 0,818 / 0,897 / 1,091 ; corrélations spectrales mid :
0,409 / 0,193 / 0,328. Les amplitudes/canaux/valeurs finies passent le
gate technique existant. Ni un collapse sonore général ni une réussite
musicale ne peuvent en être déduits. Aucune écoute humaine n'a validé
ces sorties ; statut musical_review_pending. Le champ candidate_pass du
renderer ne doit plus être lu comme une certification de qualité.

Preuves brutes :
[sonde temporelle et codes](../output/sample-expertise-pilot/ternary-quality-v7-forensics-20260924/forensics.json),
[sonde de biais](../output/sample-expertise-pilot/ternary-quality-v7-forensics-20260924/bias-forensics.json),
[mesures audio](../output/sample-expertise-pilot/ternary-quality-v7-forensics-20260924/audio-v6-control/audio_metrics.json).
Ces fichiers sous output restent locaux et ne sont pas suivis par Git ;
les mesures essentielles sont consignées dans le rapport Markdown.

### Règles durables pour la prochaine exécution

- Conserver le dernier V6 comme contrôle ; repartir du teacher préentraîné
  pour récupérer des maîtres non arrondis, sans prétendre repartir de zéro.
- Cibles et audits versionnés par code/runtime/précision, pas seulement
  par hash de poids et données.
- Sauvegarde reprenable exhaustive ; état des fenêtres fermées sur disque ;
  golden forwards avant sauvegarde ; biais vectoriels inclus.
- Journaliser transitions de codes ET distances aux seuils ; une loss qui
  varie ne prouve pas que les représentations ternaires se réorganisent.
- Pilote deux puis quatre blocs avec sortie globale, validation corrigée et
  écoute ; pas de cascade complète avec un préfixe déjà rejeté.
- Test 24 prompts encore fermé ; données vocales insuffisantes pour une
  affirmation de qualité musicale générale ; provenance PCM à renforcer.
- Rester sous 12 000 000 000 octets accélérateur, distinguer GiB/GB, profiler
  RSS/swap et durées longues ; budget disque explicite pour les maîtres.
- Cinq nouveaux tests CPU passent sur l'outil forensique. Ils ne valident
  pas les futurs correctifs V7. Aucun nouvel entraînement long lancé ici.

## 20. Résultat P3.1 et cause finale — 24 septembre 2026

Le [plan V7 révisé](TERNARY_QUALITY_RECOVERY_PLAN_V7.md) est désormais la
référence. Le warm-start P3.1 a restauré correctement les masters FP32, les
moments Adam, le pas 50 et le RNG : 224/224 paires, 0 mismatch codes/scales/
biais, round-trip cross-process passé. Il échoue pourtant sur le corpus exact
du cache : A `0,92804/0,73204`, terminal `0,75743`; P3.1
`0,92819/0,73154`, terminal `0,75873`; audio RMS hors `[0,70;1,30]` sur les
trois prompts. La perte de l'état FP32 était donc un défaut réel des anciens
runs, mais **pas la cause suffisante**.

Le diagnostic durable est maintenant :

- symmetric G32 utilise un seuil implicite trop rigide et ne réorganise que
  `0,060–0,062 %` des codes après 112 updates;
- la loss de trajectoire sur ancres P2 n'améliore pas le minimum ni le
  terminal, même avec couverture complète;
- prolonger à LR fixe dérive par rapport au P2; le dernier checkpoint ne doit
  plus être exporté sans sélection validation;
- un pilote learned-symmetric commencé depuis des records ternaires est encore
  rouge, ce qui interdit de confondre « seuil appris » et « bonne initialisation
  dense »;
- l'audit sur un dataset différent est une erreur méthodologique et doit être
  rejeté automatiquement par digest.

Conséquence : pas de cascade 0–3/24 blocs. Reprendre par calibration de sortie
de bloc, quantizer TTQ/LSQ à seuil et niveaux appris, transition soft-to-hard,
validation indépendante et `best_validation`.

## 21. Continuation V7 et décision V8 — 25 septembre 2026

Le [rapport de continuation V7](TERNARY_V7_CONTINUATION_REPORT_2026-09-25.md)
consigne les essais supplémentaires. Le meilleur candidat bloc 0 atteint
`0,96620 / 0,82497` en velocity mean/min et environ `0,85219` au terminal,
mais `release=false`. Les variantes hidden-state, dense direct, seuils,
focus prompt, composition de modules et polish rollout n'ont pas franchi le
gate. Aucun modèle complet n'a été promu.

Le diagnostic est plus précis que « il faut plus de steps » :

- la STE bouge des scales/seuils sans résoudre correctement les décisions de
  codes discrètes ;
- le master dense préserve l'information mais sa projection initiale TTQ
  détruit trop le comportement ;
- le warm-start dequantifié préserve le candidat mais ne restaure pas le
  master perdu ;
- la moyenne masque un pire prompt et le rollout ne prédit pas le forward
  reserialisé ;
- la mémoire respecte parfois 12 GB, donc elle n'est pas la cause suffisante.

Le plan actif est désormais [V8](TERNARY_QUALITY_RECOVERY_PLAN_V8.md) :
contrat de validation indépendant, statistiques d'activations time-aware,
solveur discret depuis les masters denses, reserialization après chaque
proposition, sélection Pareto et cascade bloc par bloc. Les travaux audio
récents sur la quantification time-aware, ainsi que QuEST, HadaNorm et TQ-DiT,
motivent la méthode d'activation/sensibilité ; ils ne constituent pas une
preuve que Stable Audio 3 peut être converti directement en ternaire strict.

### P0/P1/P2 V8 déjà exécutés

Le contrat externe V8 a trouvé une incohérence réelle dans le cache V7 : le
répertoire train déclaré contenait des symlinks vers plusieurs corpus, et le
contrat embarqué résolvait certains chemins vers `broad-music` ou
`universal-dataset`. Le contrat corrigé couvre `434/33/41` échantillons
train/validation/test, digest
`1d2bd975f6dad798ac105bbbf29ce42135b7daea19a3c4ee78e3776a6cc04130`, zéro
chevauchement latent/parent.

Le profil time-aware de 256 états du bloc 0 confirme que `ff.ff.2` et
`cross_attn.to_out` changent fortement avec le sigma, tandis que
`cross_attn.to_kv` est presque constant. Le premier solveur activation-aware
V8 a ensuite été rejeté après forward réel et validation : il passe de
`0,96946/0,89541` à `0,86667/0,49690` en velocity train sélectionné, puis
`0,84760/0,66785` en validation et `0,44607/0,28554` au terminal. Cette
preuve ferme la voie « MSE de poids pondérée seule » ; la suite doit scorer
chaque proposition sur la sortie de bloc et un mini-held-out avant toute
composition.

Le filtre bloc→full-DiT confirme cette règle : `ff.2` passe de
`0,95854/0,93147` à `0,98217/0,96097` au bloc, mais tombe à
`0,95741/0,77588` sur le DiT complet. `self_attn.to_out` et leur composition
baissent aussi la moyenne ou le minimum. Le score local ne doit donc jamais
être publié comme une amélioration de modèle.

## 22. V9 — corrections vérifiées et nouvelle voie (25 septembre 2026)

> Première révision V9. Les choix de périmètre et de méthode sont remplacés
> par la section 23 ; les corrections factuelles restent valides.

Le [registre détaillé](TERNARY_V9_EVIDENCE_2026-09-25.md) contient les preuves
et limites. Il prévaut sur les interprétations trop fortes de la section 21.

- **Symlinks** : les 434 latents train sont volontairement des liens. Les
  16 exemples du contrat cache V7 appartiennent tous aux cibles du train.
  Leur résolution vers plusieurs corpus ne prouve pas une fuite. La
  disjonction réelle des parents nécessite toujours un audit de filiation.
- **Held-out** : les audits V8 déclarent `heldout=false` malgré un contrat
  corpus valide. Le mécanisme doit être unifié. Les ensembles servant aux
  choix de méthode/checkpoint sont du dev, pas un test scellé.
- **Couverture** : le profil V8 de 256 états contient uniquement
  `real_latent_noised`, aucune trajectoire teacher/student. Un nombre de
  prompts × sigmas × seeds ne garantit donc pas la bonne distribution.
- **Mesures de bloc** : quatre rejeux denses du cache FP16 donnent une erreur
  L2 relative de 0,058–0,070 %. Ce petit écart n'explique pas à lui seul
  l'échec. Le calcul de `T_lat` dans le banc V8 emploie toutefois le mauvais
  axe : 1472 au lieu de 128. Le chemin bloc seul ne démontre pas l'effet
  de cette erreur sur les scores historiques.
- **Format** : TTQ avec centre et scales asymétriques n'est pas `s*q`.
  V9 apprend un déplacement pour affecter les codes mais n'ajoute aucun
  centre au poids final. Le teacher dense original reste le master.
- **Causalité** : la MSE diagonale V8 ne contrôle ni les covariances ni la
  fonction du suffixe. Son échec ne condamne pas toute méthode activation-aware.
  Inversement, un gain sans flips peut être réel si les scales améliorent
  les sorties rechargées.
- **Ressources** : les champs `metal_*_gb` historiques sont des GiB. Le
  profil V8 a utilisé 3 955 470 120 octets. G32 intégral représente toujours
  455 818 272 octets théoriques hors headers ; aucun artefact de qualité à
  cette taille n'est livré. Seulement 9,4 GiB libres au contrôle V9.

La recherche récente et ses limites figurent dans le plan : CAT-Q motive une
nouvelle calibration progressive et des fenêtres recouvrantes, mais son code
officiel consulté ne fournit pas encore le trainer, ses résultats concernent
les LLM et ne garantissent pas SA3 sous 12 GB. Bonsai Image conserve des
supports FP16 ; la rétention de benchmarks de Bonsai 2 n'est pas une
probabilité de succès du présent projet.

Ordre actif : fermer P0 → témoins et couverture → pilote strict sur deux
seeds → progression cumulative avec réouverture des fenêtres → toutes les
matrices → test scellé et écoute brute. Maximum trois rounds de pilote par
campagne bornée. Pas de nouvelle cascade tant que le pilote dur rechargé ne
passe pas. Pas de succès annoncé sans écoute humaine.

Au moment de cette section historique, le travail V9 était documentaire et
diagnostique. La section 24 consigne l'exécution pilote ultérieure ; aucun
poids teacher ni fichier de corpus n'a été modifié.

## 23. V9.1 — décision Bonsai et cible 550–650 Mo (25 septembre 2026)

L'utilisateur précise : « le but est de ternariser comme bonzai pas plus »,
puis confirme le compromis d'environ 550–650 Mo plutôt que la limite de
500 Mo. Le toutes-matrices n'est plus l'objectif ni une phase ultérieure.

Périmètre : sept matrices attention/FFN par bloc, 24 blocs, soit 168 matrices
et 1 358 954 496 éléments (93,5038 % de l'inventaire). Conserver les autres
tableaux à leur précision native : conditionnement, modulation, entrées et
sorties, convolutions, memory tokens, normes, biais, gates et buffers.
Les 6,5 % de supports de SA3 ne doivent pas être artificiellement réduits
pour reproduire un pourcentage annoncé sur une architecture différente.

Le cœur reste réellement ternaire symétrique avec une scale FP16 par groupe ;
ce changement de périmètre n'autorise pas TTQ asymétrique, offset de groupe,
LoRA cachée ou accès au teacher à l'inférence. Le paquet contient lui-même
ses supports ; text encoder et codec sont comptés séparément.

Lecture des headers NPZ, sans chargement GPU : G128 représente
550 196 256 octets, G64 571 429 920, G32 613 897 248, hors conteneur.
Ces nombres supposent une seule scale stockée ; un second champ affine
nécessaire au kernel doit être dérivé ou compté explicitement. L'ancienne
valeur 455,8 Mo appartient au périmètre toutes-matrices et ne décrit pas
le compromis désormais demandé. Voir le registre, section 9.

Ordre actif : banc fiable → baseline dense → pilotes B128/B32 sur entrée,
milieu, sortie et deux blocs → conversion cumulative du cœur → export
autonome → test réservé et écoute. Les supports restent figés. Un échec
de calibration sur le train autorise un test de masters entraînables dans
la fenêtre active ; il ne déclenche pas une nouvelle cascade aveugle.

Les similarités velocity/terminal servent à diagnostiquer, pas à attribuer
une note musicale. Les anciens seuils de cosinus sont retirés du verdict
qualité avant tout entraînement V9. Le rejet V8 n'est pas effacé.
Le skill audio-models a motivé la séparation format/technique/écoute, la
baseline native et l'absence de mastering réparateur.

L'intervalle subjectif 10–30 % est retiré : les essais ne permettaient pas
ce niveau de précision. Aucun remplacement par >90 % sans preuve.
La V9 initiale est archivée ; la révision documentaire initiale n'avait pas
encore produit de modèle ni de résultat musical. La section 24 décrit les
résultats du pilote lancé ensuite.

## 24. V9.1 — le pilote exécuté et le vrai verrou (25 septembre 2026)

La phase d'exécution a maintenant produit une preuve pilote, mais pas encore
un modèle complet. Le point important est la séparation suivante :

1. le format Bonsai est faisable sur le teacher SA3 réel : 168 matrices cœur,
   supports inchangés, G128 550,2 MB hors conteneur et G32 613,9 MB ;
2. le forward/reload est fiable : parité dense, cache replay et records
   cross-process passent ;
3. la qualité du DiT complet dépend des états de diffusion rencontrés, pas
   seulement de la reconstruction locale d'un bloc ;
4. la qualité musicale finale n'est pas déductible du cosinus : une écoute
   A/B doit encore valider le canari.

### Leçon expérimentale principale

La loss locale de bloc donne une confiance trompeuse. Les runs full-DiT G32
sur [0,1] passent avec mean/min velocity 0,96492/0,88516 et
0,96193/0,86794 sur deux seeds. Le bloc 2 pointwise tombe à
0,95662/0,84431, donc sous le minimum release. En réinjectant des états
student de production pendant quatre pas, le bloc 2 remonte à
0,95735/0,85615. La prochaine cascade doit donc conserver une voie
on-policy bornée par fenêtre, pas uniquement des cibles teacher sur états
réels bruités.

### Leçon mémoire

Le rollout on-policy 4 pas sur une fenêtre d'un bloc reste sous 12 Go
(pic observé 7,85 Go). La même idée G128 sur deux blocs a atteint 12,33 Go
et a été arrêtée par la garde. Il faut mesurer chaque fenêtre, ne jamais
désactiver la garde, et préférer une fenêtre active d'un bloc ou un stitching
réduit plutôt qu'un graphe complet qui dépasse la machine.

Un bug de seed dans la préparation des rollouts a aussi été corrigé avant le
run final : `repeats` était utilisé hors portée. Les cibles on-policy ont été
reconstruites après correction ; un cache produit avant cette correction ne
doit pas être réutilisé.

### Leçon audio

Les canaris crop 128 sont finis, stéréo, 44,1 kHz et techniquement valides.
Les meilleurs écarts appariés observés sont audio cos 0,96555 pour G32 [0,1]
seed 1 et 0,94665 pour le bloc 2 on-policy ; le seed 2 de [0,1] est plus
variable à 0,93518. Ces chiffres justifient une écoute, pas une livraison.
Un canari lancé avec crop 256 a été retiré de la comparaison : la longueur
d'entrée n'était pas celle du cache de référence.

### État de décision

Le choix courant est **G32 + supports natifs + `W=s*q` + on-policy seulement
quand le gate de trajectoire le demande**. Le pilote reste
`musical_review_pending`. Ne pas lancer les 24 blocs, ne pas exporter un
modèle final et ne pas annoncer une probabilité de réussite avant :

- écoute A/B brute sur les canaris seed 1/seed 2 et bloc 2 ;
- gel de la recette et d'un budget mémoire par fenêtre ;
- puis cascade [0,1] → [1,2] → … avec canary et rollback à chaque jalon.

Les artefacts et métriques complets sont listés dans
[`TERNARY_V9_EXECUTION_REPORT_2026-09-25.md`](TERNARY_V9_EXECUTION_REPORT_2026-09-25.md).

## 25. V10 — format Bonsai terminé, qualité cumulative rejetée (25 septembre 2026)

La cascade complète G32 a finalement été exécutée jusqu'au bloc 23 après
libération contrôlée du disque. Le bloc 23 avait d'abord échoué avec `ENOSPC`:
les checkpoints cumulatifs recopiaient le cœur à chaque bloc et les essais
abandonnés du bloc 6 occupaient plusieurs Go. Les fichiers expérimentaux et
les checkpoints intermédiaires redondants ont été retirés; la reprise depuis
le bloc 22 a donné un round-trip exact.

Le package autonome est :
[`final-ternary-bonsai-g32-v10`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/final-ternary-bonsai-g32-v10/).
Il contient 168 matrices ternaires, 357 supports natifs, 1 358 954 496 codes,
des scales positives FP16 et les modes directs/Hadamard par matrice. Taille
NPZ réelle : 513 342 833 octets; payload logique : 613 897 248 octets; sous
l'enveloppe 650 Mo.

La vérification en processus neuf (`verify_ternary_bonsai_package.py`) est
exacte : 732 paramètres comparés, zéro mismatch, forward L2 relatif 0 et
cosinus 1. Le packing et le runtime ne sont donc plus le verrou.

Le verrou restant est la dérive cumulative du champ de vitesse : mean cosine
0,876934, minimum 0,646171, état terminal moyen 0,844483, canary audio
0,852056. `technical_pass` est vrai, mais les gates candidate/release sont
fausses. Le flag `--promote-on-quality-fail` explique pourquoi la cascade a
continué; il ne constitue pas une acceptation.

Conclusion opérationnelle : le package est maintenant un **livrable local
accepté pour étude personnelle**, après écoute utilisateur. Les gates
automatiques restent publiés et ne sont pas requalifiés artificiellement en
pass. Une optimisation P1/P2 n'est nécessaire que pour une validation
générale/aveugle ultérieure. Voir
[`TERNARY_QUALITY_RECOVERY_PLAN_V10.md`](TERNARY_QUALITY_RECOVERY_PLAN_V10.md).
