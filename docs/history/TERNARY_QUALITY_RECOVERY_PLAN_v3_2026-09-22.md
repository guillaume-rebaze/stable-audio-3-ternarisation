# Plan v3 — obtenir un DiT ternaire de qualité, preuves avant entraînement

Date : 22 septembre 2026. Statut : v3 rejeté ; continuation v4 autorisée
exécutée ; aucun modèle validé.

Ce plan remplace les recommandations du [plan v2 archivé](archive/TERNARY_QUALITY_RECOVERY_PLAN_v2_2026-09-22.md). Les résultats et limites sont conservés dans le [registre des expériences](TERNARY_RECOVERY_EVIDENCE_2026-09-22.md) et la [base de connaissances](TERNARY_DISTILLATION_KNOWLEDGE_BASE.md).

## Décision en bref

Ne pas relancer la cascade actuelle. Elle comporte encore des écarts entre entraînement et exécution ; ses évaluations réutilisent des données d’entraînement. Augmenter les steps, réduire les groupes ou augmenter le rang LoRA ne répond pas d’abord à ces problèmes.

Ordre retenu :

1. Prouver le contrat numérique et corriger les mesures trompeuses.
2. Figer une référence FP16, des données séparées et un protocole audio stéréo.
3. Mesurer la sensibilité réelle des couches, puis sélectionner un opérateur ternaire symétrique.
4. Distiller sur les entrées réellement rencontrées par l’étudiant, avec checkpoints complets et validation hors entraînement.
5. Valider les trajectoires, puis l’audio natif et l’écoute.
6. Étendre le périmètre ternaire et réduire la taille sans perdre les validations précédentes.

Résultat attendu du protocole : soit un candidat vérifié dans les contraintes, soit une impasse précisément localisée. Aucun article ni plan ne garantit à l’avance la qualité de Stable Audio 3 Medium ternaire sous 12 GB.

## 1. Contrat de réussite

### 1.1 Ce que signifie « ternaire »

Distinctions obligatoires :

- **Codes ternaires** : `q ∈ {-1, 0, +1}`. C’est une propriété des codes, pas une preuve sur tout le modèle.
- **Poids ternaires symétriques par groupe** : `W_hat = s_g * q`, avec scale partagé positif. Les valeurs exécutées sont `{-s_g, 0, +s_g}`, pas littéralement `{-1, 0, +1}`.
- **Trois niveaux affines centrés** : `W_hat = m_g + s_g * q`. Le code zéro donne `m_g`, généralement non nul. C’est la méthode actuelle ; ne pas la présenter comme une ternarisation symétrique.
- **Ternaire en base tournée** : les codes sont ternaires dans une base orthogonale ; le poids équivalent dans la base d’origine est généralement dense. Rotation déclarée dans le nom et le manifeste.
- **Hybride** : une matrice dense, une LoRA ou une branche résiduelle continue reste dans le chemin annoncé ternaire. Autorisé comme contrôle, pas comme réalisation silencieuse de la cible stricte.

Deux jalons, sans confusion :

1. `core-symmetric-ternary` : les 168 projections attention/FFN sont ternaires. Les autres paramètres restent explicitement inventoriés. Jalon de recherche, pas objectif final.
2. `DiT-matrices-ternary` : toutes les matrices apprises du DiT sont couvertes, y compris conditionnement, entrées/sorties, convolutions et embeddings matriciels. Les biais vectoriels, normes, gates, scales et buffers sont déclarés séparément, avec leur précision et leur coût.

Le second jalon ne signifie toujours pas « 100 % de tous les scalaires ternaires ». Une exigence littérale incluant normes, gates, biais et scales demanderait un autre contrat et une recherche distincte. Ne jamais employer « 100 % » sans dénominateur exact. Le text encoder et SAME-L ne sont pas inclus dans la taille du DiT ; publier aussi la taille du pipeline complet.

### 1.2 Contraintes et unités

- Cible finale DiT : fichier déployable **≤500 000 000 octets** ; 455,8 Mo reste une ambition non validée.
- Publier aussi les octets des tenseurs décompressés, la mémoire résidente et les dépendances. Un NPZ compressé n’est pas son empreinte GPU.
- Limite entraînement : **12 000 000 000 octets** de mémoire accélérateur ; garde interne conservatrice **11 000 000 000 octets**. Journaliser octets, GB décimaux et GiB, sans les mélanger.
- Sur Apple Silicon : mesurer aussi RSS, pression mémoire et évolution du swap. Ne pas additionner aveuglément RSS et Metal, qui peuvent compter les mêmes pages.
- Qualité native : aucun EQ, compresseur, débruiteur ou limiteur pour masquer le modèle. WAV float32 brut conservé ; seule copie d’écoute à gain constant et sécurité true-peak.
- Aucun GPU payant, publication de poids ou changement du moteur de production dans cette phase.

Une baseline de recherche peut dépasser 500 Mo. Elle ne passe pas le contrat final pour autant. Toute relaxation de taille, de précision ou de périmètre doit rester une décision explicite.

## 2. Pourquoi les essais précédents ont échoué

### 2.1 Faits établis

- Ancienne voie Pure/H128 : 168 projections sous QAT contre 192 à l’export ; absence de chemin Hadamard correspondant au reload. L’activation effective du flag H128 dans l’artefact historique n’est pas documentée.
- Nouvelle voie : `hard_freeze_block()` écrit `max_dense_reconstruction_error = 0` via `metrics.append(0.0)`. Cette valeur n’est pas une mesure.
- `quantize_weight_mx()` reconstruit avec statistiques FP32 ; le packer sérialise scales/biais FP16. L’équivalence de ces deux forwards n’est pas prouvée.
- `model_params_for_export()` convertit les autres paramètres en FP16, y compris `timestep_features.freqs`, construit en FP32 dans le runtime.
- Sonde légère du 22 septembre : différence relative des features sin/cos de **0,8004 / 0,6242 / 0,3970** aux sigmas **0,95 / 0,50 / 0,10** après cette conversion. C’est un défaut de fidélité numérique mesuré, pas encore la part causale de l’échec audio.
- Reload group64/group32 : erreur relative maximale **0,2153 % / 0,2489 %**, malgré un cosinus proche de 1. Le seuil du script est passé à 2 %, alors que le plan demandait 0,1 %. Cause non isolée ; contrat non fermé.
- Cascade : conditionnement local précalculé depuis le teacher en FP16 ; forward complet : zéros locaux construits par défaut en FP32. Comparer les chemins avant de supposer leur équivalence.
- Entraînement principal : 193 latents, huit prompts, crop 128 ; audit de velocity sur des exemples du même corpus. Changer la seed du bruit ne crée pas un jeu held-out.
- Schedule d’entraînement à huit pas : plus petit sigma non nul proche de 0,274. La zone 0,10 annoncée n’est pas couverte par cette grille.
- Le renderer nommé `audio_mono()` sélectionne le canal gauche, pas la stéréo ni une moyenne mono. Le contrôle audio principal porte sur un seul prompt piano, une seed et environ 11,889 s.
- Group64 contre group32 : nombre de steps, polish et loss terminale changent aussi. Le gain observé n’isole donc pas l’effet du groupe.

Sources : [trainer](../services/musicgen/train_ternary_quality.py), [contrat](../services/musicgen/ternary_contract.py), [audit](../services/musicgen/audit_ternary_quality.py), [renderer](../services/musicgen/render_ternary_quality.py), [registre et fichiers de mesure](TERNARY_RECOVERY_EVIDENCE_2026-09-22.md).

### 2.2 Ce que les résultats permettent de conclure

Le group32 atteint un cosinus velocity moyen/minimum de **0,8475/0,7130** ; l’adapter rank8 sur latents réels monte à **0,8682/0,7476**, mais son contrôle audio ne passe pas. Aucun candidat n’est validé.

La dérive cumulative, le choix du quantizer, la distribution d’entraînement et la sensibilité terminale restent des explications plausibles. Elles ne sont pas séparées des écarts logiciels. Remettre le bloc 23 en FP16 ne suffit pas dans l’essai réalisé ; cela ne prouve ni une cause unique ni une impossibilité du ternaire.

Le pic de 29,42 GiB concerne une implémentation QAT globale avec poids maîtres FP32 et Adam. Il ne démontre pas que toute optimisation globale est impossible sous 12 GB. Les fenêtres à 7,78 GiB et adapters à environ 5,95 GiB prouvent seulement une faisabilité partielle côté allocateur Metal.

## 3. Recherche vérifiée et décisions retenues

Sources primaires consultées le 22 septembre 2026 ; recherche ciblée, non revue exhaustive. Résultats publiés par leurs auteurs, non reproduits localement.

- [TerDiT, version révisée avril 2025](https://arxiv.org/html/2405.14854v2) : QAT ternaire de DiT image entraînés depuis zéro ; modification de l’AdaLN avec normalisation. Décision : instrumenter modulations et résidus. Ne pas greffer une nouvelle norme sans ablation ; aucune preuve de conversion rapide de SA3 sous 12 GB.
- [PTQ for Audio Diffusion Transformers, septembre 2025](https://arxiv.org/abs/2510.00313) : Stable Audio Open, W8A8/W4A8, calibration temporelle et compensation low-rank. Décision : contrôler la couverture des timesteps et garder un baseline 4-bit diagnostique si nécessaire. Ce n’est ni du ternaire ni Stable Audio 3.
- [BitNet Distillation, octobre 2025](https://arxiv.org/abs/2510.13998) : conversion de LLM avec warm-up, SubLN et distillation des relations d’attention. Décision : préserver les états maîtres et tester les signaux auxiliaires séparément. Les résultats sur tâches de langage ne fixent pas nos pertes audio.
- [CAT-Q, juin 2026](https://arxiv.org/abs/2606.26650) : modulation apprise et transition différentiable vers la ternarisation, avec code annoncé disponible. Décision : challenger conditionnel pour l’optimisation des seuils. Ne pas transposer le nombre de données ou le coût GPU annoncé au DiT audio.
- [Bonsai Image 4B, fiche du modèle](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-unpacked) : modèle image et formats low-bit distincts du format dense dépacké. Décision : séparer qualité de la fonction, format de stockage et performance du kernel. La fiche n’établit pas une recette d’entraînement SA3.
- [Bonsai 2, runtime officiel](https://github.com/PrismML-Eng/Bonsai-demo) et [format du modèle](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/README.md) : rotation des activations nécessaire ; plusieurs packings et coûts effectifs. Décision : versionner et tester la base de rotation, refuser un runtime incompatible. « g128 » ne signifie pas automatiquement « H128 ».
- [Self Forcing, juin 2025, révision novembre](https://arxiv.org/abs/2506.08009) : écart train/test traité sur vidéo autorégressive. Décision par analogie, à tester : interroger le teacher sur les états du student. Notre sampler audio n’est pas autorégressif ; ce n’est pas une reproduction de Self Forcing.

La recherche soutient des expériences précises, pas une promesse « group16 + logits = qualité ». Aucun résultat consulté ne certifie notre combinaison exacte modèle, données, taille et mémoire.

## 4. Architecture numérique à construire

### 4.1 Un snapshot, plusieurs exécutions équivalentes

Créer un contrat versionné `TernarySnapshot` : codes entiers, scales à la précision déployée, shape originale/paddée, groupe, biais de couche éventuel, convention de rotation, dtypes des buffers et hashes.

Le snapshot alimente le forward de référence, le packer, le reload et le préfixe gelé. L’export n’appelle plus un autre quantizer pour redéfinir les poids. Garder séparément le checkpoint d’entraînement avec poids maîtres et état optimiseur.

Première voie stricte : `W_hat = s_g * q`, sans moyenne de groupe apprise et sans rotation. Groupe 64 pour isoler le changement par rapport aux essais ; groupe 128 comme challenger de coût, à budget identique.

Initialisation proposée, non optimum supposé :

1. Initialiser `s_g` depuis les poids du teacher ; traiter explicitement les groupes nuls.
2. Alterner assignment `q = clip(round(W/s_g), -1, 1)` et ajustement positif `s_g = sum(W*q)/sum(q²)` pendant un petit nombre d’itérations borné.
3. Choisir l’initialisation sur l’erreur de sortie de couche `||X Wᵀ - X W_hatᵀ||²`, avec activations train couvrant les timesteps ; pas uniquement sur l’erreur des poids.
4. Pour la QAT, utiliser les valeurs et arrondis du déploiement dans le forward ; backward par estimateur explicitement testé. Les poids maîtres restent continus et sauvegardés.

Conserver l’ancien `m+s*q` comme contrôle nommé `affine-3level`, jamais comme réalisation de cette voie stricte.

### 4.2 Compatibilité MLX et définition du zéro

Avec codes affine `c = 1-q`, la convention MLX peut représenter le cas symétrique via `scales=-s`, `biases=+s`, donc `scales*c+biases=s*q`. Le `biases` du format n’est pas un offset appris indépendant. Le biais vectoriel du `Linear` reste un autre tenseur.

Contraintes : code 3 interdit ; scales/biases exactement opposés à la précision stockée ; le code zéro doit déquantifier vers zéro exact. Vérifier les groupes nuls, sous-flux FP16, padding, transpositions, dimensions non divisibles et groupes non supportés. Ne pas imposer un scale non nul à un groupe nul par une convention implicite.

Stockage initial : slots 2-bit, compatible avec le backend installé après tests. Cela ne donne pas 1,58 bit physique par poids.

### 4.3 Hadamard : challenger, pas prérequis

Pour activations en lignes et `RᵀR=I` : `W_r=W R`, `W_hat=Q Rᵀ`, `y=(x R) Qᵀ`. Tester ce sens, la normalisation, les signes et le padding avec un oracle FP32 avant toute QAT.

Conserver séparément `rotation_block_size` et `quant_group_size`. Une matrice d’entrée 257 exige padding des activations ET des poids, puis slicing vérifié ; aucun token de conditionnement supprimé.

Si retenu : manifeste de rotation, opérateur dédié, tests dense/packé, gradients et benchmark de latence. Un fichier qui charge dans un `QuantizedLinear` ordinaire n’est pas une preuve que la rotation est appliquée. Les poids effectifs en base d’origine ne doivent pas être qualifiés de ternaires symétriques.

## 5. Étapes bloquantes avant le premier nouveau pilote

### G0 — figer l’état et les contraintes

Travail :

- Conserver les essais rejetés ; n’en écraser aucun. Répertoire neuf `output/sample-expertise-pilot/ternary-quality-v3/<run_id>/`.
- Enregistrer révision Git, patch des fichiers modifiés/non suivis pertinents, Python, MLX, runtime, OS, GPU et commandes. Le commit seul ne décrit pas ce worktree.
- Hasher poids teacher, text encoder, codec, manifests et scripts ; distinguer checkpoint BASE/ARC réellement utilisé et sampler compatible.
- Inventorier toutes les matrices, buffers, paramètres non quantifiés, formes et octets. Calculer le budget final avant de choisir une extension de périmètre.
- Figer `scope_train`, `scope_export`, `scope_reload`, `scope_inference` ; égalité vérifiée, pas un prédicat vague sur les noms.
- Vérifier disque disponible, mémoire et exclusivité d’un seul gros job MLX. Préserver une réserve disque d’au moins 15 GB ; chiffrer checkpoints/caches avant allocation.

Sortie : `environment.json`, `scope.json`, `size_budget.json`, `run_config.json`. Passage : aucun élément requis inconnu. Durée d’un futur run extrapolée après mesure, jamais déduite d’un ancien nombre de steps.

### G1 — corriger et tester la fidélité logicielle

Fichiers : `ternary_contract.py`, `train_ternary_quality.py`, loaders d’audit et de rendu. Tests proposés dans `services/musicgen/tests/`, **à créer**, pas déjà passés.

1. Remplacer la métrique zéro fictive par une comparaison calculée. Une mesure absente vaut `null` avec statut incomplet, jamais zéro.
2. Préserver les dtypes contractuels : fréquences Fourier et autres buffers déterministes, normes, gates, conditionnements. Supprimer le cast FP16 global.
3. Comparer le modèle gelé avant export et le reload, paramètre ET buffer par paramètre/buffer : noms, shape, dtype, valeurs. Vérifier aussi les attributs non sérialisés.
4. Auditer le chargement du teacher : mapping de clés contrôlé ; seules les clés supplémentaires/manquantes justifiées par une allowlist versionnée sont acceptées. `strict=False` sans rapport interdit.
5. Tester code pack/unpack exact, code invalide, fichier corrompu, mauvais groupe, mauvais scope/hash, scale non fini. Le loader doit refuser ces cas.
6. Comparer oracle dense déquantifié, forward QAT gelé, forward packé et reload. Couvrir couches réelles, groupes 32/64/128, crops 32/128/324, dtypes supportés, plusieurs seeds et amplitudes.
7. Comparer `DiT.__call__` et exécution découpée aux mêmes frontières : preprocessing, texte, temps, local condition, tokens mémoire, sorties de blocs, projection et postprocessing. Le student emploie son propre conditionnement dès que celui-ci est entraîné ou quantifié.
8. Tester reprise : sauvegarde/reload avant un pas puis exécution du pas suivant avec mêmes RNG et état optimiseur. Déquantifier un NPZ puis réinitialiser Adam n’est pas une reprise.

Tolérances distinctes :

- Codes, scope, valeurs sérialisées et dtypes : exacts.
- Pack/unpack et reconstruction du même snapshot par le même oracle : exacts ; calcul FP32 de référence `rtol=1e-6`, `atol=1e-7` comme seuils initiaux lorsque les opérations diffèrent.
- Modèle packé contre son reload, mêmes kernels/dtypes : viser identité ; tolérer uniquement le bruit de répétabilité mesuré. Tout changement de buffer est bloquant, même avec un excellent cosinus.
- Dense contre kernel packé : enveloppe numérique calibrée par couche, précision et amplitude, avant optimisation. Alerte à `1e-3` d’erreur L2 relative ; dépassement exige diagnostic et seuil versionné, pas relâchement pour sauver un run.
- Les tests incluent des sorties proches de zéro : erreur absolue en plus de la relative et du cosinus.

Passage : `contract_report.json` complet, aucune différence inexpliquée. Refaire l’audit des anciens group32/group64 après correction dans des artefacts dérivés distincts, sans QAT, pour mesurer la part du logiciel. Ne pas remplacer les originaux.

### G2 — référence et données sans fuite

Référence : DiT FP16 courant sans adapter, même text encoder, SAME-L et sampler que le déploiement prévu. Garder le checkpoint de style accepté séparément ; ne pas changer simultanément quantification et style.

Trois contrôles : teacher répété à seed identique ; teacher même prompt/seeds différentes ; aller-retour codec sur sources autorisées. Ils mesurent respectivement répétabilité, diversité normale et plancher codec. Écouter au moins une sortie teacher représentative avant de le choisir comme cible.

Données :

- Manifeste source/latent/caption exact, hashes conteneur et PCM, codec/révision, gains, offsets, durées, parent et droits par usage. Captions fournies conservées comme telles ; elles ne prouvent pas ce qui est entendu.
- Les 193 latents historiques restent `debug/train_seen`. Aucun reclassement rétroactif en test final des exemples déjà optimisés ou consultés pour sélectionner un modèle.
- Séparer par enregistrement parent/session, pas par crop. Dédoublonner les groupes de filiation avant le split. Interdire les dérivés d’un même parent dans train et validation/test.
- Pilote train proposé : 256–512 états/crops issus d’au moins 64 parents autorisés et 16 conditionnements distincts, si disponibles ; sinon déclarer couverture insuffisante, ne pas fabriquer de diversité par duplication.
- Validation : au moins 24 parents inédits et 12 prompts tenus à l’écart. Test final : autres parents et 24 prompts figés avant sélection ; trois seeds par prompt, durées 12 et 30 s. La taille définit un pilote, pas une preuve universelle.
- Couverture à documenter : piano/acoustique, percussions/transitoires, basse/sub, textures longues, arrangements denses, voix si dans le périmètre, styles réellement présents. Ne pas annoncer une couverture absente.
- Nouvelles sources ou états teacher générés : provenance, droits et acceptation technique avant usage ; aucun téléchargement massif ni corpus externe supposé autorisé.

Passage : hashes de splits, absence de fuite, baseline reproductible et références stéréo disponibles. Si le corpus n’autorise pas cette séparation, produire seulement un smoke test ; pas de nouvelle déclaration de qualité générale.

## 6. Expériences d’identification et entraînement

### G3 — identifier où naît l’erreur

Avant une nouvelle cascade, produire un profil de sensibilité sur les mêmes entrées :

1. Remplacer une famille de matrices ou un bloc à la fois dans une copie contrôlée du teacher FP16 ; conserver le teacher de référence intact. Mesurer sortie locale, velocity et estimation débruitée.
2. Ajouter les blocs quantifiés progressivement ; mesurer aussi chaque préfixe exécuté par le vrai forward.
3. Aux points sensibles, comparer `student_block(h_teacher)` et `teacher_block(h_teacher)` : erreur locale à entrée identique.
4. Comparer `teacher_block(h_student)` et `teacher_block(h_teacher)` : effet de la dérive amont avec bloc inchangé.
5. Comparer `student_block(h_student)` et `teacher_block(h_student)` : erreur locale sur distribution étudiante.
6. Mesurer séparément branche résiduelle, modulation, tokens audio et tokens mémoire. Un excellent cosinus sur la somme résiduelle peut cacher une mauvaise branche transformée.

Vérifier notamment blocs 0, 12, 22, 23, projections de conditionnement et sortie finale. Pas de conclusion « bloc 23 responsable » issue d'une seule restauration FP16. Le contrôle de restauration doit garder les autres paramètres, entrées et dtypes identiques.

Sortie : `sensitivity.json`, attribution par couche/timestep et choix d’une intervention. Une quantification 4-bit du même scope peut servir de contrôle : si elle échoue aussi, examiner d’abord pipeline/données/objectifs. Elle n’est pas un livrable ternaire.

### G4 — sélectionner un seul contrat candidat

Comparer au plus quatre configurations ; le contraste principal ne change qu’une variable :

- A : symétrique direct, groupe 64, initialisation calibrée sur activations.
- B : même configuration, groupe 128 ; évalue le compromis taille/qualité.
- C : même groupe que A, seuil/scale appris avec transition douce puis codes durs ; inspiration CAT-Q, adaptation à valider, pas reproduction annoncée.
- D : même groupe que A, rotation orthogonale explicite ; uniquement si l’opérateur passe G1 et si G3 montre un problème compatible avec les outliers.

Groupe 32 affine historique reste un témoin, pas un bras directement comparable. Un nouveau témoin affine 64 peut remplacer un challenger si l’on veut isoler précisément le centrage. Group16 et kernel personnalisé ne sont pas prioritaires : ils augmentent métadonnées et travail logiciel sans garantie de gain.

Budget de sélection proposé : blocs 0, 12 et 23 ; 100 updates par bloc/configuration, deux seeds d’optimisation, mêmes données et même ordre. Jusqu’à 250 updates seulement pour les deux meilleurs bras si la validation continue de progresser. Évaluer les codes durs, pas la seule relaxation douce.

Enregistrer : erreur de sortie/velocity, ratios de norme, occupation des trois codes, fraction de changements de codes, dérive des scales, gradients, temps et mémoire. Un taux de changement nul n’est pas un échec en soi ; couplé à une validation bloquée, il signale une piste d’optimisation à revoir.

Choix : amélioration hors entraînement reproductible, pire bucket non dégradé, contrat et mémoire conformes. Ne pas lancer quatre cascades complètes. Si aucun bras ne progresse, revenir au diagnostic ou déclarer la méthode non concluante.

### G5 — distillation progressive avec objectif de sortie

#### Distribution d’entraînement

Employer un mélange initial documenté, à tester :

- 1/3 d’interpolations de latents réels autorisés ;
- 1/3 d’états de trajectoires teacher ;
- 1/3 d’états de trajectoires student.

Pour chaque état `x`, cible principale **`teacher(x, t, condition)` sur ce même état**. Ne pas comparer la velocity du student sur sa trajectoire à celle du teacher sur une autre entrée en prétendant mesurer une erreur de prédiction appariée.

Les rollouts utilisent le sampler et le schedule de livraison : grille 12 pas pour le profil pilote, puis 24 pas uniquement si ce second profil fait partie du contrat. Enregistrer tous les bruits de re-noise, pas seulement la seed initiale. Garder teacher/student à bruits appariés pour le diagnostic.

Ajouter aux interpolations réelles une couverture stratifiée de `[0,02; 0,20]`, `(0,20; 0,50]`, `(0,50; 0,80]`, `(0,80; 1]`, plus les points exacts du sampler. Les états cachés de rollout conservent leur vrai timestep ; ne jamais leur attribuer artificiellement un autre sigma. Compter les observations par bucket et par prompt.

Les proportions sont une hypothèse de départ. Tester le bénéfice des états student contre une baseline teacher-only au même nombre de cibles et d’updates. Rafraîchir le cache student au début de chaque passe et après chaque 200 updates ; indexer chaque état par checkpoint source. Limiter le cache par budget explicite, pas en gardant toutes les activations en mémoire.

#### Objectif principal

Pour la convention locale `v` et `z_hat = x - t*v`, validée contre le sampler :

```text
NMSE(a,b) = mean((a-b)^2) / max(mean(b^2), energy_floor)
L_velocity = NMSE(v_student, v_teacher)
L_direction = 1 - cosine(v_student, v_teacher)
L = L_velocity + 0.1 * L_direction
```

`energy_floor` est fixé sur train pour éviter les explosions près du silence. Calculs de pertes en FP32. Moyenne équilibrée par exemple/bucket, pas une seule corrélation sur tous les tokens concaténés.

Mesurer aussi `z_hat`, dont l’erreur peut devenir importante quand le débruitage soustrait des termes proches, ainsi que RMS/DC par canal. Ne pas additionner d’emblée six pertes non calibrées. Ajouter un terme RMS, temporel ou spectral seulement après ablation identifiant son utilité ; la dérivée temporelle des latents n’est pas une mesure directe du spectre audio.

#### Initialisation séquentielle

1. Préfixe étudiant hard-quantifié, rechargé et gelé ; teacher sans gradient.
2. Bloc courant QAT avec poids maîtres préservés. Optimiseur limité à une allowlist ; pas de `block.unfreeze()` englobant des paramètres inutilisés.
3. Perte locale audio-token sur la sortie teacher correspondante ; signal mémoire distinct et pondéré. Le diagnostic G3 distingue cible sur entrée teacher et cible sur entrée student.
4. À chaque checkpoint local, exécuter le suffixe réel et mesurer la velocity globale. Une amélioration locale avec dégradation de sortie n’est pas acceptée.
5. Choisir le meilleur checkpoint de validation, produire son snapshot dur, passer G1 ; ce snapshot devient le préfixe suivant.

Départ proposé : batch 1, accumulation 4, crop 128, AdamW sur le seul bloc courant, LR `2e-5`, clipping gradient `1.0`, weight decay `0` pour scales/normes/gates. C’est une configuration de pilote, pas un hyperparamétrage universel. Seconde LR possible `5e-5`, une seule variable changée. Préciser que les steps ci-dessous sont des updates optimiseur, pas des microbatches.

Budget : 200 updates/bloc, checkpoint et validation toutes les 50 ; extension jusqu’à 400 seulement si la validation progresse encore. Audit de sortie après les blocs 0, 2, 5, 11, 17 et 23. Si le pire bucket chute de plus de 0,02 de cosinus ou si sa NMSE augmente de plus de 10 % par rapport au checkpoint précédent, suspendre l’extension et analyser. Ces limites sont des garde-fous initiaux, pas des seuils perceptuels.

#### Correction globale par fenêtres

Optimiser ensuite 1 bloc, puis 2, puis au maximum 4 blocs contigus **selon le profil mémoire mesuré**, avec perte de velocity en sortie du DiT complet. Les autres poids sont gelés, mais le suffixe conserve le gradient par rapport à ses entrées : geler les paramètres ne signifie pas détacher les activations.

Ordre des fenêtres issu de G3 ; au plus deux passes complètes. Départ : 100 updates/fenêtre, checkpoint toutes les 25. Conserver les états maîtres/optimiseur ; ne pas rouvrir depuis les seuls poids ternaires déquantifiés. Lors d’un changement de snapshot amont, invalider les caches dépendants.

Cette étape entraîne aussi les scales si l’opérateur fournit un gradient correct, avec arrondi de déploiement dans le forward. Les normes/gates restent dans l’allowlist annoncée. Les branches LoRA ne sont pas ajoutées à la voie stricte.

#### Rollout court, seulement si nécessaire

Après amélioration de la prédiction sur états student, comparer loss un-pas et loss deux-pas au même budget. Quatre pas uniquement après mesure mémoire. Utiliser mêmes états initiaux/bruits et teacher gelé ; le gradient traverse les pas student réellement optimisés. Un cache de 32 états teacher suivi d’une MSE un-pas n’est pas une loss de rollout différentiable.

Une loss audio via SAME-L gelé reste optionnelle, sur fenêtres aléatoires adaptées au contexte du décodeur, après succès du contrôle latent. Décrire contexte et padding ; ne pas toujours décoder le premier préfixe. Vérifier son coût avant l’activation ; aucun changement de codec pour améliorer artificiellement un score.

### G6 — passage au périmètre final et réduction de taille

Après qualité `core-symmetric-ternary` validée, traiter les familles hors core une à une, dans l’ordre de sensibilité et de gain d’octets mesuré : conditionnement local/global/texte/temps, entrées-sorties, convolutions, embeddings matriciels. Les 168 matrices ne couvrent pas à elles seules le DiT.

Chaque extension reprend G1, une calibration/QAT ciblée, audit latent, rendu court et validation. Si une famille casse la qualité, conserver le dernier candidat accepté avec son nom `core` ou `hybrid` ; ne pas la laisser FP16 en continuant à annoncer le périmètre final.

Optimisations de taille, dans cet ordre :

1. Retirer les états maîtres/optimiseur du paquet d’inférence, pas des checkpoints de recherche.
2. Éviter les métadonnées redondantes : biais affine dérivable de la scale dans le contrat symétrique, reconstruction contrôlée au chargement.
3. Tester groupe 128 ou allocation de groupes par couche, sous budget et à qualité constante. Les groupes doivent apparaître couche par couche dans le manifeste.
4. Tester un packing de trits plus dense seulement si nécessaire ; pack/unpack exact et kernel/latence vérifiés. Cinq trits par octet donnent 1,6 bit/code avant scales et padding, pas exactement `log2(3)`.
5. Mesurer fichier compressé, tenseurs chargés, pic de décompression et temps d’inférence. Ne jamais citer la seule taille ZIP comme empreinte de calcul.

Le changement de packing ne doit pas changer `q` ou `s`. La réduction de groupe et les rotations changent la fonction : elles exigent une nouvelle validation qualité.

## 7. Validation qui peut réellement autoriser une livraison

### G7 — trajectoires et audio stéréo

#### Rendu et mesure

- Évaluer uniquement le fichier rechargé et, le cas échéant, les dépendances dont le hash est enregistré.
- Préserver les deux canaux. Mesurer chacun, puis Mid/Side, corrélation stéréo et compatibilité mono ; pas de sélection silencieuse du canal gauche.
- Déduire longueur latente du codec/runtime. Avec le facteur local 4096 à 44,1 kHz, 128 tokens donnent environ 11,889 s. Douze secondes demandent au moins 130 tokens, trente au moins 323, puis arrondi aux contraintes du runtime et recadrage audio final exact. Vérifier les durées réellement écrites.
- Conditionnement de durée, crop et rendu doivent correspondre. Pas de champ `seconds=30` posé sur un latent de 129 tokens.
- Mesurer avant gain : finitude, crête, plateaux d’écrêtage, RMS/LUFS, true peak, DC, silence, ruptures, enveloppes et bandes fréquentielles. Un float brut au-dessus de 1 n’est pas à lui seul un fichier écrêté ; il impose une copie d’écoute protégée.
- Copies d’écoute à niveau égal, gain constant documenté et true-peak ≤−1 dBTP ; conserver les gains et les bruts. Pas de normalisation dans les métriques de fidélité.

#### Trois axes, aucun score unique

1. **Fidélité au teacher.** Cosinus et NMSE de velocity, erreur de `z_hat`, dérive des trajectoires à chaque pas, ratios RMS/DC. Mesurer par prompt, seed, timestep et durée ; publier moyenne, médiane, p05/p95 et pire cas.
2. **Qualité et adéquation.** Spectres multi-résolution/log-mel, énergie des bandes, transitoires, bruit, cohérence stéréo ; pertinence audio-texte avec évaluateur externe figé si disponible. CLAP reste un proxy, pas un juge musical.
3. **Diversité.** Différences inter-seed à prompt fixé, absence de copies et effondrement de diversité, comparées au teacher. Une imitation au sample près n’est pas requise pour être musicalement utile.

Les distances spectrales appariées servent à mesurer la fidélité sur sorties comparables ; une faible corrélation peut aussi accompagner un autre arrangement correct. L’ancien seuil universel `spectral_correlation ≥0.85` est retiré : ni le choix `log1p(PSD)` ni ce seuil n’ont été calibrés perceptuellement. Un FAD sur quelques fichiers est également insuffisant ; reporter effectif, embedding et incertitude si un benchmark distributionnel est ajouté.

#### Gates automatiques provisoires

Pour proposer une écoute d’acceptation, viser sur held-out : cosinus velocity moyen ≥0,93 et minimum ≥0,85 sur les cas préenregistrés. Ce sont des filtres d’ingénierie, pas une certification audio ; l’écoute diagnostique de rejets reste utile à tout moment. Ajouter les erreurs absolues/relatives, car le cosinus ignore le gain.

RMS student/teacher hors `[0,70; 1,30]` : alerte à examiner avec le contenu ; jamais réparée par normalisation des mesures. NaN, silence involontaire, plateau d’écrêtage créé par export ou canal manquant : rejet technique immédiat.

Calibrer les autres seuils avant sélection sur teacher répété, autres seeds et dégradations contrôlées : bruit, coupure fréquentielle, silence, clipping. Figer ensuite `acceptance.json`. Toute modification ultérieure crée une nouvelle version du protocole et exige de réévaluer les candidats ; le test final n’est pas utilisé pour régler les seuils.

#### Taille des validations et écoute

Progression : deux prompts de validation × deux seeds × 12 s ; puis 12 prompts × trois seeds × durées 12/30 s. Seul le meilleur candidat passe au test final scellé : 24 prompts × trois seeds × deux durées, soit 144 rendus par modèle pour un profil de sampler. Teacher mis en cache une fois. Un autre nombre de pas ou une autre durée constitue un autre profil à tester.

Écoute aveugle randomisée à niveau égal sur sous-ensemble stratifié choisi avant connaissance des scores : au moins 24 paires, contrôles teacher/teacher et quelques répétitions. Évaluer artefacts, transitoires, grave, aigus, stéréo, respect du prompt et utilité musicale.

Pour une conclusion de non-infériorité : proposer au moins trois auditeurs, marge préfixée de 5 points sur 100, intervalle de confiance groupé par prompt/auditeur. Si l’incertitude traverse la marge, résultat inconclusif ; ne pas le transformer en succès. Une seule acceptation par Guillaume vaut validation d’usage locale documentée, pas preuve statistique générale.

Statuts séparés : `technical_pass`, `musical_review_pending`, `accepted_local`, `rejected`, `inconclusive`. La génération ne peut pas se déclarer elle-même musicalement acceptée.

### G8 — ressources et paquet final

Profilage indépendant de chaque phase : chargement, caches de conditionnement, teacher, forward, backward, optimiseur, checkpoint/export/reload, sampler et décodage. Synchroniser les opérations MLX avant mesure ; reset de pic seulement aux frontières documentées, conserver le maximum global.

Stratégie sous contrainte : batch 1, accumulation, teacher sans gradient, text encoder déchargé après cache, décodeur absent pendant QAT latent, activations rematérialisées et fenêtres de paramètres entraînables. Offload et optimiseur à états réduits restent des alternatives à profiler, pas des économies présumées.

Tester d’abord 10 updates à la plus petite forme, puis 50 à la plus grande forme autorisée. Estimer la mémoire avant élargissement. Si la garde de 11 milliards d’octets est atteinte, interrompre sans nouvel export massif, conserver le dernier checkpoint et réduire le profil. Une mesure après chaque pas ne peut empêcher une allocation transitoire trop grande : le préflight doit donc rester conservateur.

Sur Mac : journaliser RSS/Metal/swap avant et après, pression et concurrence ; une hausse persistante du swap interdit de conclure « tient sous 12 GB ». Pas de preuve CUDA 12 GB à partir du seul Metal.

Paquet final : artefact rechargé, manifests, hashes, scope complet, dtypes, code/runtime compatibles, taille disque et tenseurs, profil mémoire, latence, rapport held-out, WAV témoins et verdict d’écoute. Un paquet sans rapport d’acceptation reste expérimental.

## 8. Backlog d’implémentation, dans l’ordre

Cette liste était le backlog initial. Les livrables P0/P1/P2 marqués dans le
rapport d’exécution existent désormais ; les gates qualité et données restent
cependant ouvertes.

1. **P0 — contrat et buffers.** Modifier `ternary_contract.py` et `train_ternary_quality.py` ; ajouter `tests/test_ternary_contract.py` et `tests/test_ternary_reload.py`. Critère : G1, notamment fréquences temporelles inchangées, erreur réellement mesurée et validation des arrays NPZ au reload.
2. **P0 — forward découpé.** Factoriser conditionnements/préfixes partagés ; ajouter `tests/test_ternary_forward_parity.py`. Critère : mêmes sorties intermédiaires que `DiT.__call__`, mémoire/audio distingués, zéro-step sans dérive.
3. **P0 — auditeur bloquant.** Modifier `audit_ternary_quality.py` et `render_ternary_quality.py` ; ajouter tests de canaux/durée et fixtures de rejet. Critère : NaN, scope erroné, fichier manquant ou gate échoué donnent code de sortie non nul ; aucun JSON non standard avec NaN.
4. **P1 — splits et configuration.** Ajouter `prepare_ternary_dataset.py`, `configs/ternary_quality_v3.json` et tests de filiation. Critère : G0/G2, configurations rejettent clés inconnues et paramètres incompatibles, aucun fallback silencieux.
5. **P1 — checkpoints.** Persister poids maîtres, quantizer, optimiseur, compteur, RNG Python/NumPy/MLX, ordre des données et hashes. Critère : reprise équivalente au pas suivant ; snapshot de livraison immuable, distinct du checkpoint d’entraînement.
6. **P1 — sensibilité et fenêtres.** Ajouter audit de G3 ; corriger `finetune_ternary_window.py`. Critère : gradient traversant le suffixe gelé, zéro-step sans re-quantification destructive, garde mémoire et sélection du meilleur checkpoint.
7. **P2 — états student.** Ajouter collecte/caches de trajectoires et objectif apparié de G5. Critère : teacher évalué sur le même `x`, trace du sampler et des bruits, cache invalidé quand ses dépendances changent.
8. **P2 — pilote puis cascade.** Exécuter G4, puis G5 seulement si les résultats l’autorisent. Aucun parallélisme de gros jobs sur le Mac.
9. **P3 — périmètre, compression, livraison.** Exécuter G6–G8 ; mettre à jour le registre et la fiche modèle, sans changer le moteur de production avant acceptation.

### Commandes et état réel

Depuis la racine du dépôt, ces commandes existent déjà :

```bash
rtk proxy python3 services/musicgen/ternary_contract.py
rtk proxy python3 services/musicgen/train_ternary_quality.py --help
rtk proxy python3 services/musicgen/audit_ternary_quality.py --help
rtk proxy python3 services/musicgen/render_ternary_quality.py --help
```

Le contrat, le reload, la parité forward, le dataset et le checkpoint sont
testés par les tests ciblés. Le loader d’audit revalide aussi les codes/scales
du NPZ avant instanciation MLX. La configuration v3 refuse les clés inconnues
et le préparateur de dataset refuse un run annoncé held-out quand les sources
indépendantes manquent.

Commande de tests de fidélité exécutée :

```bash
rtk proxy python3 -m pytest -q \
  services/musicgen/tests/test_ternary_contract.py \
  services/musicgen/tests/test_ternary_reload.py \
  services/musicgen/tests/test_ternary_forward_parity.py
```

La suite ciblée élargie du 22/09/2026 couvre aussi checkpoint et configuration
dataset : `12 passed`. Le meilleur snapshot a été ré-audité après ajout de la
validation des arrays NPZ ; le contrat passe, la quality gate reste rejetée.

Le point d’entrée de cascade et la fenêtre G5 ont été exécutés avec export
atomique et reload exact. Le préflight held-out bloque encore correctement la
livraison ; aucun résultat rejeté ne doit écraser le runtime de production.

## 9. Budget, arrêt et reprise

### État d’exécution du 22 septembre 2026

Le run G0–G8 et les deux fenêtres contrôlées sont documentés dans
[`TERNARY_V3_EXECUTION_REPORT_2026-09-22.md`](TERNARY_V3_EXECUTION_REPORT_2026-09-22.md).
Le contrat/reload passe ; le meilleur snapshot reste rejeté sur velocity/audio.
La reprise optimizer/RNG complète est maintenant implémentée et testée sur une
reprise courte. La distillation multi-pas student réelle est également
instrumentée, mais son pilote n’améliore pas la velocity. Le split indépendant
v4 est désormais construit et testé ; la qualité reste rejetée sur validation
et test, donc aucune claim de qualité générale n’est autorisée.

Audit local complémentaire : `broad-music/latents-12s` contient 120 latents et
`sftminimal/latents-12s` 58. Leur métadonnée native reste
`source_annotation_status=provided_unreviewed`, sans licence déclarée ;
l’autorisation explicite de Guillaume les rend utilisables pour étude personnelle
locale uniquement. Les 46 sources broad recouvrant le train v3 sont exclues du
held-out ; les 132 sources restantes sont réparties par parent dans le split v4.
La sélection et l’autorisation sont conservées dans
`g2-authorized-expanded-v3/selection_manifest.json`.

Budget initial proposé :

- Sélection G4 : maximum quatre bras courts ; deux seulement prolongés ; deux seeds d’optimisation, aucune recherche exhaustive de learning rates.
- Cascade G5 : une configuration choisie, une seed d’abord ; seconde seed seulement si la première valide les critères. 200 updates/bloc, maximum 400 sur preuve de progrès.
- Fenêtres : deux passes maximum ; exécutées le 22/09 (`20–23` rollout puis
  `23` conservatrice), toutes deux rejetées ; aucune troisième passe v3.
- Chaque run long : plafond initial de quatre heures ou limite d’updates, première atteinte. Mesurer le débit après chauffe ; si le plan ne tient pas, réduire le round et sauvegarder, pas prolonger implicitement.
- Audio : petits contrôles avant validation large ; test final une fois sur le candidat sélectionné. Échec final : nouvelle version de recherche, test désormais considéré consulté.
- Stockage : réserver le dernier checkpoint, le meilleur et les preuves nécessaires avant lancement. Jamais de suppression automatique des originaux, modèles rejetés ou résultats d’un autre run.

Arrêt immédiat : non-finitude, mismatch de scope/buffer, baseline inconnue, corruption, absence de données autorisées, dépassement mémoire, pression/swap persistants, disque insuffisant.

Arrêt de l’hypothèse : aucune amélioration de validation pendant trois checkpoints, ou deux expériences contrôlées consécutives sans bénéfice conjoint. Ne pas augmenter simultanément steps, rang, groupe et loss pour éviter ce verdict.

Branches possibles si l’objectif strict reste hors d’atteinte :

- **Bug identifié** : corriger puis réévaluer sans entraînement ; quantifier ce que la correction change.
- **Données insuffisantes** : demander/constituer un corpus autorisé réellement nouveau ; pas de répétition déguisée en diversité.
- **Optimisation bloquée** : tester un unique changement de quantizer/seuil/reprise, avec diagnostic des codes et gradients.
- **Capacité ternaire insuffisante sur familles précises** : proposer soit périmètre hybride déclaré, soit nouveau student ternaire natif nécessitant architecture, données et budget distincts. Ne pas livrer l’hybride comme l’objectif strict.
- **Mémoire seulement** : réduire fenêtre/contexte ou profiler recomputation/optimiseur ; ne pas relancer automatiquement la QAT globale qui a atteint 29,42 GiB.
- **Qualité bonne, taille trop grande** : G6 à codes inchangés d’abord ; rapporter l’écart si la limite reste dépassée.

## 10. Mémoire de projet et définition de terminé

Chaque run conserve au minimum :

```text
run_config.json        environment.json       code.patch
scope.json             dataset_manifest.json  splits.json
contract_report.json   sensitivity.json       acceptance.json
checkpoints/           snapshots/             manifests/
velocity_metrics.json  trajectory_metrics.json
audio_metrics.json     listening_report.json  memory_metrics.json
size_budget.json       run_summary.json       train.log
raw-stereo/            listening-copies/
```

Les noms sont une convention cible. Tous les fichiers ne sont pas nécessaires à une sonde isolée, mais leur absence interdit les claims qui en dépendent. Hashes et chemins des preuves dans `run_summary.json`, dates, durée, échecs et raison d’arrêt inclus. Écriture atomique, validation avant promotion, aucun statut global `success` par simple fin de boucle.

États d’exécution : `preflight_pass`, `contract_pass`, `pilot_complete`, `exported`, `technical_pass`, `musical_review_pending`, `accepted_local`, `rejected`, `inconclusive`, `blocked`. Un artefact peut être exporté et rejeté.

Checklist finale :

- [ ] Scope final exact ; codes et scales conformes ; paramètres non ternaires explicités.
- [ ] Entraînement, snapshot, reload et runtime concordants ; buffers préservés.
- [ ] Splits indépendants et droits d’usage documentés.
- [ ] Qualité de trajectoire validée sur données non utilisées pour optimiser.
- [ ] Audio stéréo natif valide à 12/30 s et profils de sampler annoncés.
- [ ] Écoute et incertitude documentées ; aucun défaut masqué par mastering.
- [ ] Fichier ≤500 000 000 octets, ou candidat clairement hors cible ; tailles décompressées publiées.
- [ ] Pic sous contrainte démontré sur machine cible, pas estimé depuis un autre backend.
- [ ] Checkpoint reprenable, paquet rechargé en processus neuf et preuves référencées.
- [ ] Base de connaissances et registre mis à jour, rejets conservés.

**État final v3 : expérimental/rejeté.** Le meilleur snapshot est
`output/sample-expertise-pilot/ternary-quality-v3/window-terminal-g64-safe/`;
il ne doit pas remplacer le runtime de production. La suite exige d’abord un
corpus indépendant autorisé, puis une nouvelle hypothèse de capacité (scope,
groupe ou student ternaire natif) ; ne pas relancer la même QAT sous un autre
seed.

### Continuation v4 autorisée — résultat

Le run v4 a utilisé le train élargi `253` samples / `202` parents, avec
validation `18` prompts / `28` parents et test `24` prompts / `66` parents.
Le candidat G64 strict exporté fait `479 660 087` octets et recharge exactement,
mais obtient `0,81248 / 0,65934` en validation et `0,80827 / 0,69551` au test.
Le contrôle audio brut validation échoue `3/3`. Cette hypothèse est donc
rejetée ; aucune promotion ni troisième QAT identique n’est autorisée.

Prochaine hypothèse acceptable : augmenter la capacité effective avec un
student ternaire natif, un scope hybride explicitement annoncé ou un résidu de
contrôle séparé. Toute variante doit conserver le split v4 scellé et repasser
G1–G8.
