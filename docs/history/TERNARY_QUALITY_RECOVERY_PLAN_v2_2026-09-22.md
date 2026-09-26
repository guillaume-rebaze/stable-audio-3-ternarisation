# Archive — plan de récupération v2, 22 septembre 2026

Document historique, non exécutable en l’état. Remplacé par
[le plan v3](../TERNARY_QUALITY_RECOVERY_PLAN.md).
Les affirmations ci-dessous « contrat/reload correct », « cause non liée au
packing » et « seule voie 100 % ternaire » ne sont pas établies. Les seuils
audio, le périmètre ternaire et les unités mémoire sont également corrigés
dans le plan v3. Les liens locaux ont été adaptés au dossier d’archive ; le
contenu historique est conservé ci-dessous.

**Date :** 22 septembre 2026  
**Contexte :** échec du run Pure Ternary/H128 et de l’artefact 456 MB.  
**Objectif :** obtenir d’abord un modèle ternaire rechargé et acoustiquement crédible, puis réduire la taille.  
**Contrainte :** pic d’entraînement <12 GB VRAM ; gate interne <11 GB.

## Décision

Ne pas relancer un entraînement avec plus de steps sur les scripts actuels.

L’échec est d’abord un échec de contrat logiciel et de validation :

- le forward QAT, l’export et le reload n’utilisent pas la même quantification ;
- le périmètre des couches entraînées diffère du périmètre exporté ;
- H128 est annoncé mais absent du chemin d’inférence ;
- les métriques locales de bloc masquent l’erreur de trajectoire ;
- le script Pure Ternary ne fait pas le polish E2E annoncé ;
- l’audit audio normalise les WAV et arrive après l’export, trop tard pour arrêter le run.

Le nouveau plan doit prouver chaque prérequis sur un petit test avant de consommer plusieurs heures de QAT.

## Résultat de l’exécution — 22 septembre 2026

Le nouveau contrat et le reload sont corrects, mais le premier modèle strictement
`core-ternary` n’est pas livrable. Les mesures sont faites sur le fichier rechargé,
avec les WAV écrits sans normalisation :

| Run | Taille | Velocity moyen / min | Audio 12 s | Verdict |
|---|---:|---:|---:|---|
| group64, cascade + polish | 512,63 MB | 0,8076 / 0,6780 | spec 0,1618 ; RMS 1,17× | échec |
| group32, cascade + polish | 580,27 MB | 0,8475 / 0,7130 | spec 0,2408 ; RMS 1,61× | échec |
| group32, rollout fenêtre 20–23 | 580,29 MB | 0,8127 / 0,6924 | spec 0,2279 ; enveloppe 0,5762 | échec |
| group32 + adapter rank8 rollout | +34,02 MB | 0,8442 / 0,7367 | spec 0,1548 ; RMS 0,79× | échec |
| group32 + adapter rank8 réel | +34,01 MB | 0,8682 / 0,7476 | spec 0,1515 ; RMS 1,50× | échec |

Les trois runs ont bien rechargé 168 projections ternaires. Le group32 est la
meilleure granularité disponible dans le `QuantizedLinear` MLX installé ; group16
est refusé par le runtime. Le reload contractuel passe (cosinus ≥0,999997), mais
la trajectoire complète ne passe pas (gate candidat : velocity moyen ≥0,90,
minimum ≥0,80, audio spectral ≥0,85).

Le profilage explique l’échec : les blocs 0–22 restent relativement proches,
alors que le préfixe accumule une dérive avant le bloc terminal ; remettre le
bloc 23 en FP16 ne remonte le cosinus moyen qu’à 0,8528. Une QAT E2E de toutes
les matrices monte à 29,42 GB ; la version fenêtre de quatre blocs reste à
7,78 GB, mais ne gagne que 0,003 de cosinus. Le problème restant est donc la
représentation et la distribution d’entraînement, pas l’export NPZ ni la VRAM.

Conclusion : aucun fichier group64/group32 ne doit être présenté comme modèle
ternaire de qualité. Les artefacts sont des diagnostics reproductibles.

## Plan v2 après échec du run

### Voie de livraison qualité sous 12 GB — prochaine voie à implémenter

La première tentative base group32 + adapter rank8 a tenu 5,95 GB mais a échoué
les gates audio : entraîner uniquement sur les états teacher ne suffit pas.
La prochaine tentative doit donc utiliser un adapter rank16/32 avec DAgger :
alterner états teacher et états générés par le student, puis optimiser
velocity + spectral loss + enveloppe sur plusieurs seeds. Le runtime doit
charger un `TernaryAdapterLinear` déclaré dans le manifeste ; il ne faut pas
fusionner l’adapter puis re-quantifier silencieusement. Cette variante n’est
pas « 100 % ternaire », mais c’est la route réaliste à tester pour obtenir
qualité et mémoire sous 12 GB ; son nom et sa taille doivent le déclarer.

Gates de cette voie : reload exact de la base, pic <11 GB, velocity moyen ≥0,90,
minimum ≥0,80, spectral ≥0,85, RMS entre 0,70× et 1,30×, puis validation 30 s.

### Voie 100 % ternaire — recherche, avant toute nouvelle promesse de taille

1. Écrire un opérateur/kernel group16 (ou codebook appris) puisque le runtime
   affine MLX ne l’accepte pas actuellement.
2. Remplacer le STE à seuil fixe par des logits de codes ternaires avec
   température/annealing et scale+bias appris par groupe.
3. Générer des paires teacher/student sur rollouts sampler, pas seulement sur
   des interpolations de latents réels.
4. Faire la distillation par fenêtres de quatre blocs sous 11 GB, puis un
   passage de calibration des codes avec les sigmas 0,95/0,75/0,50/0,25/0,10.
5. Rejeter automatiquement tout checkpoint qui échoue à la velocity ou à
   l’audio brut ; ne mesurer la cible de taille qu’après ces gates.

Cette voie est la seule qui puisse justifier « 100 % ternaire », mais elle
nécessite un opérateur différent du `nn.QuantizedLinear` affine courant.

## 1. Post-mortem du dernier échec

### 1.1 Résultat réel

Artefact : [dit_medium_bonsai_ternary_456mb.npz](../../output/sample-expertise-pilot/universal-models/dit_medium_bonsai_ternary_456mb.npz)

- taille réelle : 386 758 441 octets, environ 368,8 MiB ;
- 906 clés NPZ ;
- 192 préfixes quantifiés ;
- 192 couches contiennent weight/scales/biases ;
- aucun manifeste de run associé au fichier ;
- aucun log ne prouve si H128 était activé.

Rendus : [bonsai_ternary_summary.json](../../output/sample-expertise-pilot/bonsai-ternary-renders/bonsai_ternary_summary.json)

| Prompt | Peak brut | RMS brut | Corrélation spectrale |
|---|---:|---:|---:|
| piano | 0,0411 | 0,00654 | 0,0688 |
| funk | 0,8626 | 0,09736 | 0,1241 |
| ambient | 0,0299 | 0,00464 | 0,2867 |

Le renderer remonte le peak à 0,85 avant d’écrire le WAV. Cette normalisation masque le niveau absolu ; elle ne répare ni la trajectoire ni le spectre. Le diagnostic doit toujours conserver les métriques brutes.

### 1.2 Ce que les anciens logs prouvaient déjà

[distill_bonsai.log](../../output/sample-expertise-pilot/distill_bonsai.log) rapporte des CosSim de blocs de 0,99+, mais son audit E2E descend à 0,1735–0,2306 autour de sigma 0,5. Le score de bloc n’est donc pas un critère de qualité.

Le script [distill_bonsai_full_500mb.py](../../services/musicgen/distill_bonsai_full_500mb.py) audite aussi un étudiant QAT en mémoire. Un score avant export ne valide pas le fichier réellement rechargé.

## 2. Causes racines, par priorité

### P0 — le modèle exporté n’est pas le modèle entraîné

Dans [train_bonsai_pure_ternary.py](../../services/musicgen/train_bonsai_pure_ternary.py) :

- le QAT H128 optionnel appelle quantize_weights avec rotation ;
- le flag --use-hadamard est désactivé par défaut, et l’artefact n’a aucun log pour prouver l’option utilisée ;
- freeze_and_quantize_block remplace ensuite le poids par une matrice dense quantifiée ;
- export_bonsai_pure_ternary re-quantifie cette matrice avec une règle brute ;
- les codes et scales calculés pendant le QAT ne sont jamais sauvegardés ;
- le reload utilise nn.quantize affine standard, sans opérateur H128.

Si H128 était activé, la rotation optimisée est donc perdue ou re-quantifiée dans une autre base. Même sans H128, une seconde quantification après le freeze peut changer les codes et scales.

**Correction obligatoire :** quantize_group retourne une structure unique q, scale, bias, convention et scope. Cette structure est utilisée par le forward QAT, le snapshot de bloc, le packer, le reload de référence et l’inférence.

### P0 — scope QAT/export différent

Le QAT Pure remplace 7 projections par bloc, soit 168 matrices :

- self-attention : to_qkv, to_out ;
- cross-attention : to_q, to_kv, to_out ;
- FFN : ff.0.proj, ff.2.

L’export parcourt toutes les matrices compatibles sous transformer.layers. Il exporte 192 matrices, donc ajoute notamment to_local_embed.seq.2. Cette couche n’a pas été entraînée sous QAT dans le script Pure.

to_local_embed.seq.0, d’entrée 257, reste FP16 car son entrée ne divise pas par 128. project_in/out, l’embedder global, les normes et les gates restent aussi hors scope.

**Correction obligatoire :** le scope est une donnée de manifeste. Le run échoue si scope_train != scope_export != scope_reload.

### P0 — H128 n’existe pas dans le runtime d’inférence

Un poids tourné W H peut être ternarisé dans l’espace Q, mais le calcul exige alors une transformation H de l’activation ou un kernel équivalent. Le renderer actuel fait seulement :

- création d’un DiT standard ;
- nn.quantize bits=2/group_size=128 ;
- load_weights ;
- multiplication affine standard.

Aucun H128 n’est appliqué à l’activation. H128 ne peut donc pas être déclaré dans le modèle final tant qu’un HadamardTernaryLinear et son export/reload ne sont pas intégrés.

**Décision :** première voie de qualité sans H128, avec quantification directe exactement compatible MLX. H128 devient une voie expérimentale séparée, jamais un flag silencieux.

### P1 — données et objectif trop faibles

Le script Pure :

- ne calibre que les 32 premiers latents ;
- utilise essentiellement le premier conditionnement prompt dans la cascade ;
- entraîne sur 12 s puis évalue surtout 30 s ;
- n’exécute pas le polish E2E annoncé dans son docstring ;
- ne sauvegarde pas de seed/config/manifest de run ;
- mesure la perte sur tout le hidden state, donc les 64 tokens mémoire peuvent dominer les tokens audio.

Le nombre de steps ne peut pas réparer ces divergences.

### P1 — préfixe précédent pas toujours hard-quantifié

Dans les scripts Bonsai group64/v5, student.freeze puis s_block.unfreeze entraîne le bloc courant, mais les blocs précédents restent des modules QAT avec poids maîtres continus. L’export final re-quantifie ensuite ces poids.

La cascade n’a donc pas toujours vu le préfixe réellement présent dans le fichier final.

**Correction obligatoire :** après chaque bloc, écrire q/scales, reconstruire le bloc depuis ces valeurs, puis utiliser ce bloc gelé et rechargé comme préfixe de l’étape suivante.

### P1 — absence de gate avant rendu long

Aucun run ne doit arriver au rendu 30 s si les tests suivants n’ont pas passé :

1. round-trip d’une matrice ;
2. round-trip d’un bloc ;
3. round-trip du préfixe déjà entraîné ;
4. velocity E2E sur 5 sigma ;
5. audio 12 s sur un seul prompt.

## 3. Architecture du nouveau pipeline

Créer une voie unique, indépendante des scripts v1–v5 :

- services/musicgen/ternary_contract.py ;
- services/musicgen/train_ternary_quality.py ;
- services/musicgen/audit_ternary_quality.py ;
- services/musicgen/render_ternary_quality.py.

Les anciens scripts restent historiques. Aucun nouveau run de production ne doit importer leur quantizer ou leur exporter.

### 3.1 Contrat de quantification direct

Première version : group_size=64, sans H128, pour obtenir une baseline de qualité avec le format affine MLX déjà disponible.

Pour chaque ligne de poids W :

1. découper les groupes ;
2. centrer si cette variante est choisie par l’ablation ;
3. calculer q dans {-1,0,+1} ;
4. calculer scale et bias ;
5. produire une référence dense déquantifiée ;
6. packer q/scales/biases sans recalcul ultérieur.

Le contrat doit fournir :

- quantize_group(W) → q, scale, bias, stats ;
- pack(q, scale, bias) ;
- unpack(pack(...)) ;
- dequantize(q, scale, bias) ;
- checksum du scope et des shapes ;
- histogramme des codes ;
- erreur de reconstruction.

Le forward QAT, le FrozenTernaryLinear de cascade et MLX QuantizedLinear de reload doivent produire la même sortie à la tolérance fixée.

### 3.2 Scope initial

Commencer par core-ternary :

- les 7 projections attention/FFN par bloc ;
- 168 matrices ;
- to_local_embed.seq.2 reste FP16 tant qu’il n’a pas son propre QAT ;
- seq.0, project_in/out, embedder global, normes et gates restent explicitement FP16.

Ce scope est honnête et permet de mesurer la contribution de chaque famille. Le modèle n’est pas appelé whole-DiT-ternary à cette étape.

Ensuite seulement :

1. QAT de seq.2 ;
2. QAT de project_in/out ;
3. QAT de l’embedder global ;
4. traitement dédié de seq.0 avec padding/slicing vérifié ;
5. changement de nom whole-DiT-ternary après inventaire complet.

## 4. Plan d’exécution par gates

### Gate A — référence FP16 déterministe

Produire une référence avant QAT :

- prompts piano, funk, ambient, techno, jazz et voix ;
- durées 12 s et 30 s ;
- seeds 0, 1 et 2 ;
- sigma 0,95, 0,75, 0,50, 0,25 et 0,10 ;
- mêmes bruit, scheduler, conditionnements et decoder pour teacher/student ;
- velocity, latent, RMS brut, peak brut, log-mel et enveloppe temporelle.

Fichier attendu : references/teacher_metrics.json.

Si le teacher n’est pas reproductible, stopper. Toute comparaison ultérieure serait fausse.

### Gate B — test synthétique du packer

Sur des matrices petites et des poids réels :

- q ne contient que -1, 0, +1 ;
- chaque scale est finie et non nulle ;
- pack puis unpack est exact ;
- forward reference et forward MLX rechargé ont une erreur relative <1e-6 ;
- les paths exportés sont exactement ceux du manifeste.

Ce test doit durer quelques secondes. Il bloque tout entraînement si échec.

### Gate C — un bloc, puis reload

Entraîner uniquement le bloc 0 sur un petit sous-ensemble.

Après chaque snapshot :

1. packer le bloc ;
2. recharger le bloc ;
3. comparer le forward QAT au forward reload ;
4. comparer les sorties audio-position et memory-position séparément ;
5. vérifier les norms, RMS et DC ;
6. arrêter si la différence reload dépasse 0,1 %.

Ne passer aux 24 blocs que lorsque ce gate est vert.

### Gate D — cascade réellement hard-quantifiée

Pour chaque bloc i :

1. teacher calcule target et trajectoire ;
2. préfixe étudiant est composé de blocs déjà packés/rechargés ;
3. bloc i est le seul bloc QAT ;
4. entraînement sur des crops et prompts multiples ;
5. sortie audio-token loss séparée des 64 tokens mémoire ;
6. snapshot q/scales/biases ;
7. reload du bloc ;
8. bloc rechargé devient le préfixe du bloc i+1.

Configuration de départ :

- 193 latents réels, tirage aléatoire ;
- au moins 10 conditionnements de genres ;
- crops de 128, 256 et 323 tokens ;
- 400 steps par bloc, batch 1 ;
- AdamW, lr initial 5e-5, decay cosinus, gradient clip 1 ;
- sigma tiré avec sur-échantillonnage de 0,10–0,50, zone où l’échec est apparu ;
- pertes hidden normalisée, cosine, RMS, DC, dérivée temporelle et velocity.

Le nombre 400 est un point de départ, pas un certificat. Arrêt si le gate de bloc ne progresse plus ou si le reload diverge.

### Gate E — distillation velocity E2E

Après la cascade :

- geler tous les q/scales/biases ;
- déverrouiller seulement les paramètres FP16 autorisés : norms, scales/shifts/gates et project_out si le scope le permet ;
- teacher et student utilisent exactement les mêmes x, t, prompt et seed ;
- optimiser la velocity à plusieurs sigma ;
- ajouter une perte de rollout court sur 4 à 8 pas ;
- re-exporter et reloader après chaque checkpoint évalué.

Ne jamais optimiser un poids maître QAT puis exporter silencieusement une autre quantification.

### Gate F — audio avant compression finale

Sur chaque checkpoint rechargé :

- 12 s d’abord ;
- 30 s ensuite ;
- 6 prompts minimum ;
- 3 seeds ;
- sans normalisation avant les métriques.

Gates candidat :

- velocity cosine moyen ≥0,90 ;
- velocity cosine minimum ≥0,80 par bucket sigma ;
- corrélation spectrale ≥0,85 ;
- RMS brut student/teacher entre 0,70 et 1,30 ;
- peak brut student/teacher entre 0,50 et 1,50 ;
- enveloppe temporelle corrélation ≥0,75 ;
- aucun rendu silencieux ou NaN.

Gates release :

- velocity cosine moyen ≥0,93 ;
- minimum ≥0,85 ;
- audio validé sur 12 s et 30 s ;
- aucun genre sous le gate candidat.

### Gate G — VRAM et taille

Mesurer aux frontières de chaque phase :

- active memory Metal ;
- peak memory Metal ;
- RSS ;
- swap ;
- teacher/student simultanés ;
- batch, crop et dtype.

Passage : pic <11 GB.

Publier simultanément :

- taille réelle du fichier ;
- taille théorique par tenseur ;
- scope quantifié ;
- bytes codes/scales/biais ;
- poids FP16 restants.

## 5. Route H128, seulement après baseline verte

H128 n’est pas la première implémentation.

### Contrat nécessaire

Pour chaque groupe :

- stocker q dans l’espace tourné ;
- au runtime, calculer activation_group × H128 ;
- appliquer q/scales ;
- reproduire exactement le forward de référence ;
- intégrer un kernel ou HadamardTernaryLinear dans le renderer ;
- mesurer coût et VRAM.

Un export standard MLX affine ne suffit pas.

### Critère de décision

- si group64 direct passe les gates audio : tester group128 direct ;
- si group128 direct passe : comparer H128 ;
- si H128 ne passe pas le round-trip ou dégrade l’audio : le retirer ;
- ne jamais sacrifier la qualité à l’étiquette H128.

## 6. Ce qui peut être livré

### Livrable 1 — ternary-quality-group64

- core-ternary ;
- format standard MLX affine ;
- scope et poids FP16 déclarés ;
- qualité audio validée ;
- taille réelle à mesurer ; avec seq.2 conservée en FP16, le fichier peut dépasser 500 MB. C’est une baseline de qualité, pas encore la cible finale de compression.

### Livrable 2 — ternary-quality-group128

- même pipeline ;
- moins de scales ;
- seulement si les gates E2E/audio passent.

### Livrable 3 — whole-DiT-ternary

- uniquement après QAT et reload de chaque matrice ;
- seq.0 traité par opérateur dédié ou maintenu FP16 avec nom honnête ;
- norms/gates documentés séparément ;
- aucun chiffre 455,8 MB avant inventaire réel.

## 7. Artefacts obligatoires d’un run

Chaque run écrit un dossier versionné :

- config.json ;
- seeds.json ;
- scope.json ;
- teacher_metrics.json ;
- layer_roundtrip.json ;
- block_metrics.json ;
- velocity_metrics.json ;
- audio_metrics.json ;
- memory_metrics.json ;
- manifest.json ;
- modèle packé ;
- log stdout complet.

Le mot complete ne doit apparaître dans le log que si tous les gates sont verts. Un fichier qui se charge mais ne passe pas l’audio est un artefact expérimental, pas un modèle réussi.

## 8. Ordre exact recommandé

1. Implémenter ternary_contract.py et le test de pack/unpack.
2. Corriger le renderer pour charger le scope du manifeste, sans prédicat implicite.
3. Passer Gate A et Gate B.
4. Entraîner et recharger un seul bloc.
5. Faire 3 blocs avec préfixe réellement packé.
6. Faire 24 blocs en group64 core-ternary.
7. Faire velocity E2E avec poids ternaires gelés.
8. Valider audio 12 s, puis 30 s.
9. Mesurer VRAM et taille.
10. Tester group128 direct.
11. Tester H128 dans un runtime dédié, uniquement si nécessaire.
12. Étendre vers whole-DiT-ternary.

## 9. Règles d’arrêt

- mismatch QAT/reload : arrêt immédiat ;
- scope différent entre train/export/reload : arrêt immédiat ;
- velocity cosine <0,80 à sigma 0,50 : retour à la dernière étape verte ;
- RMS brut <0,70 ou >1,30 du teacher : pas de rendu public ;
- corrélation spectrale <0,85 : pas de claim qualité ;
- pic VRAM ≥11 GB : réduire le batch/crop ou changer de phase, ne pas ignorer ;
- H128 sans kernel de reload : H128 désactivé.

## 10. Références utiles

- [TerDiT](https://arxiv.org/abs/2405.14854) : QAT ternaire DiT et traitement de l’AdaLN.
- [Post-Training Quantization for Audio Diffusion Transformers](https://arxiv.org/abs/2510.00313) : calibration dépendante du timestep sur un DiT audio.
- [TQ-DiT](https://arxiv.org/abs/2502.04056), [LRQ-DiT](https://arxiv.org/abs/2508.03485), [HadaNorm](https://arxiv.org/abs/2506.09932) : groupes/timesteps et rotations pour réduire les outliers.
- [Q-VDiT](https://proceedings.mlr.press/v267/feng25q.html), [MPQ-DMv2](https://arxiv.org/abs/2507.04290) : distillation de relations temporelles.
- [BitNet b1.58](https://arxiv.org/abs/2504.12285), [BitDistill](https://arxiv.org/abs/2510.13998) : poids maîtres haute précision et distillation multi-signal.
- [PrismML Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo) : nécessité d’un runtime/kernel cohérent avec le format ternaire.

## Conclusion

Le prochain succès ne viendra pas d’un run plus long du script Pure Ternary. Il viendra d’un petit pipeline vérifiable où le poids entraîné, le poids packé, le poids rechargé et le poids exécuté sont le même objet mathématique.

Priorité : **group64 direct, core-ternary, reload exact, audio validé**. Ensuite seulement : group128, H128 et whole-DiT.
