# Plan V7 — rendre la ternarisation de SA3 fiable, puis bonne à l'écoute

Date : 24 septembre 2026.
Statut : **P2, P3 et P3.1 exécutés ; tous échouent aux gates qualité ; arrêt
avant P4, aucun passage à 0–3.** Voir le
[rapport d'exécution V7](TERNARY_V7_EXECUTION_REPORT_2026-09-24.md).

Ce document est le plan V7 initial. Le plan actif est désormais
[TERNARY_QUALITY_RECOVERY_PLAN_V10.md](TERNARY_QUALITY_RECOVERY_PLAN_V10.md),
avec les preuves V9 et le package Bonsai G32 vérifié.
La [révision V7 après P3.1](TERNARY_QUALITY_RECOVERY_PLAN_V7.md) est historique.
La [V6 est conservée sans modification](archive/TERNARY_QUALITY_RECOVERY_PLAN_v6_2026-09-24.md).
Lire les [preuves du diagnostic](TERNARY_V7_FORENSIC_REPORT_2026-09-24.md)
pour les mesures complètes et la [base de connaissances](TERNARY_DISTILLATION_KNOWLEDGE_BASE.md)
pour l'historique.

## 1. Décision : réparer ce que l'on entraîne et sauvegarde avant d'augmenter le budget

Objectif inchangé : convertir le **Stable Audio 3 Medium préentraîné** en
modèle à matrices ternaires de qualité, fichier inférieur à 500 Mo décimaux,
optimisation locale sous 12 GB décimaux. Pas d'entraînement depuis zéro.

La prochaine exécution n'est pas « encore 250 updates sur le dernier V6 ».
Elle commence par rendre identiques les contrats de train, génération,
reprise et export. Ensuite seulement, un pilote sur deux puis quatre blocs
doit prouver un apprentissage utile et une amélioration après décodage audio.
Cette séquence est maintenant arrêtée au gate P3 : les résultats ci-dessous
remplacent les prévisions quand ils les contredisent ; aucune étape P4 ne
devient automatique.

Trois faits nouveaux changent la priorité :

| Défaut prouvé | Conséquence | Correction V7 prioritaire |
| --- | --- | --- |
| Aucun changement net de 226 492 416 codes entre cinq raffinements ; seules les scales changent | Prolonger les records arrondis ne prolonge pas les trajectoires des poids maîtres | Maîtres FP32 persistants, moments optimiseur et RNG sauvegardés ; transitions mesurées |
| Cibles et audit ponctuel avec t FP16, sampler avec t FP32 | L'entraînement n'approxime pas exactement la fonction utilisée pour générer | Contrat temporel unique, cibles régénérées et caches versionnés |
| Cache/audit de trajectoire réutilisait le RNG initial pour les ré-injections ; le runtime emploie `seed` initial et `seed+1` pour le sampler | Les trajectoires distillées/mesurées ne reproduisaient pas exactement l'inférence | Flux séparés, trace partagée, parité exacte contre `sample_flow_pingpong` ; cache v2 refusé |
| Biais FFN entraînables absents des records-only | L'export peut abandonner une partie de ce que le trainer a appris | Sauvegarde de tous les paramètres modifiés ; parité avant sauvegarde → reload |

Le défaut de t suffit à produire jusqu'à 9,55 % d'erreur relative teacher
sur une sonde de 24 états. Il n'explique pas seul le mauvais modèle :
à sigma=1, où l'arrondi ne change rien, le student a encore un cosinus de
velocity de 0,727 sur le prompt funk examiné.

**La réussite ne peut pas être garantie par un plan de recherche.**
V7 remplace les relances aveugles par des expériences réfutables, avec
un arrêt court si le mécanisme d'apprentissage reste inefficace.
L'objectif est un bon modèle, pas un nouveau fichier qui passe seulement
le packing ou un seuil de cosinus.

## 2. Contrat final, non négociable

### Modèle et taille

- Toutes les matrices apprises du DiT : attention, FFN, conditionnements,
  projections d'entrée/sortie, convolutions, tokens mémoire ; inventorier
  aussi le conditionneur de durée livré dans le checkpoint.
- Par groupe : q ∈ {-1,0,+1} et W = s*q, s partagé positif. Un groupe nul
  est représenté par q=0 et une convention de scale déclarée.
- Pas de moyenne affine ajoutée aux poids ; pas de code 3 réservé utilisé ;
  pas de résidu dense/LoRA permanent, ni somme de matrices contournant le contrat.
- Biais **vectoriels** de couches, normes, gates, scales et buffers conservent
  leur précision déclarée. Le biais technique du kernel doit coder exactement
  s*q et ne constitue pas une moyenne apprise supplémentaire.
- T5 et SAME-L ne sont pas ternarisés dans ce livrable. Leur disque et leur
  RAM figurent séparément dans le coût du pipeline complet.
- Sans rotation pour la voie principale. Toute variante Hadamard est nommée
  « ternaire en base tournée », jamais présentée comme W d'origine ternaire.

Inventaire lu : 230 tenseurs de dimension ≥2, 1 452 609 536 éléments.
Estimation toutes matrices G32, padding inclus : **455 818 272 octets**
(codes 363 438 080 ; scales FP16 90 859 520 ; autres tenseurs 1 520 672).
Ajouter le conteneur et les buffers nécessaires ; mesurer l'artefact final.
Cible : **≤500 000 000 octets**, pas 500 MiB.

Le dernier V6 est seulement core-G32 blocs 0–3 et pèse 2,325 Go.
Il reste un contrôle de diagnostic, pas une base de reprise exacte ni un
candidat livré. Les estimations G64/G128 historiques ne sont pas des preuves
de qualité ; changer de groupe ne répare pas un mauvais checkpoint.

### Runtime

Conserver le teacher ARC distribué, son sampler **pingpong huit pas**,
T5, SAME-L et le conditionnement de durée correspondants. Ne pas lui
substituer le sampler d'un modèle BASE ou augmenter les pas en espérant
compenser un défaut de fidélité.

Entraînement : activations FP16 ; maîtres, Adam, calculs sensibles et
réductions de loss FP32. Timesteps et fréquences Fourier FP32 sur tous les
chemins. Le mélange du sampler reproduit exactement les casts du runtime
de référence : ne pas changer simultanément son arithmétique.

## 3. Recherche récente : ce que l'on peut réellement transférer

Revue de sources primaires au **24 septembre 2026**.
Les résultats image/LLM ne sont pas des résultats SA3 audio.

| Travail | Résultat ou idée utile | Décision pour V7 |
| --- | --- | --- |
| [TRS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Scheduling_Weight_Transitions_for_Quantization-Aware_Training_ICCV_2025_paper.html), [code](https://github.com/cvlab-yonsei/TRS) | Les updates continus ne déterminent pas seuls les transitions des poids quantifiés ; distance aux seuils et rythme des transitions comptent | Journaliser flips et distances aux seuils ; n'adapter le LR qu'avec ces mesures, pas avec la loss seule |
| [QuEST, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_QuEST_Low-bit_Diffusion_Model_Quantization_via_Efficient_Selective_Finetuning_ICCV_2025_paper.html), [code](https://github.com/hatchetProject/QuEST) | DiT de génération d'images ; identifie des couches sensibles et des activations déséquilibrées, puis ajuste sélectivement les poids sous supervision locale et globale | Profilage par couche et supervision de sortie complète conservés ; utile pour choisir le prochain scope, **pas** preuve sur audio ni sur 1,58 bit |
| [ParetoQ, révision octobre 2025](https://arxiv.org/abs/2502.02631) | À très faible précision, les représentations se réorganisent davantage qu'à 3–4 bits | Conserver les maîtres et permettre un déplacement réel ; ne pas confondre préservation du teacher et proximité élémentaire de ses poids |
| [LSQ](https://arxiv.org/abs/1902.08153) | Apprentissage du pas de quantification avec gradient mis à l'échelle | Challenger simple et testable si le contrôle corrigé plafonne ; notre adaptation groupée ternaire doit être validée séparément |
| [CAT-Q, juin 2026](https://arxiv.org/abs/2606.26650), [BitTern](https://github.com/IntelChina-AI/BitTern) | Modulation/ternarisation assouplie et reconstruction par fenêtres de LLM | Inspiration pour une seconde recherche, pas une recette SA3 déjà validée. Vérifier la disponibilité du code d'entraînement avant de promettre une reproduction |
| [Tequila, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b555804d495321df2e3208cc27f4fbc-Abstract-Conference.html), [code](https://github.com/Tencent/AngelSlim/tree/tequila/TernaryQuant) | Identifie le piégeage des maîtres dans la zone morte ; réactive ces poids via un signal auxiliaire continu | Mesurer proximité du seuil et flips. Ne pas ajouter son biais dynamique à l'inférence : ce serait un autre contrat et une autre taille |
| [RobuQ, v2 mai 2026](https://arxiv.org/html/2509.23582v2) | QAT de DiT image avec rotation, mais aussi **branche résiduelle FP de bas rang** | Corriger la lecture V6 : ses résultats ne prouvent pas notre contrat strict sans résidu ; pas d'importation aveugle |
| [PTQ Audio DiT](https://arxiv.org/abs/2510.00313) | Stable Audio Open en W8A8/W4A8 ; sensibilité temporelle et aux canaux ; correction résiduelle | Justifie la couverture des timesteps et les mesures audio, pas une conversion ternaire garantie |
| [Bonsai Image, mai 2026](https://prismml.com/news/bonsai-image-4b), [artefact](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-mlx-2bit) | Cœur ternaire, mais tenseurs de support FP16, notamment conditionnement/projection de sortie | Exemple utile de déploiement ; **pas** preuve d'un DiT entièrement ternaire selon notre scope |
| [Bonsai 2 27B, 17 septembre 2026](https://prismml.com/news/bonsai-2-27b) | Nouvelle annonce et artefact de LLM ternaire de grande taille | Faisabilité externe encourageante ; métriques éditeur et architecture différentes, aucun budget SA3 déduit |
| [BITCOS, 14 septembre 2026](https://arxiv.org/abs/2609.16338) | Packing ternaire adapté à la fréquence des zéros | À considérer après la qualité ; gagner des octets ne corrige pas l'apprentissage |

Conclusion : une conversion utile est plausible, mais aucun de ces travaux
ne démontre la combinaison **SA3 Medium + toutes matrices + 500 Mo +
12 GB locaux + qualité audio conservée**. L'hypothèse de V7 porte d'abord
sur un protocole correctement exécuté, puis sur l'optimisation.

## 4. Ordre d'exécution et critères de passage

Aucune phase coûteuse ne démarre sans les preuves de la précédente.

| Phase | Travail | Preuve attendue avant la suite |
| --- | --- | --- |
| P0 | Réparer temps, état de train, export et audits | Tests de régression et roundtrips réels, dont cas négatifs |
| P1 | Figer runtime, données, contrôles et budgets | Baseline teacher reproductible ; manifests ; profils mémoire |
| P2 | Prouver l'apprentissage sur deux blocs | Hard-QAT qui progresse, reprises fidèles, validation et audio pilote |
| P3 | Corriger l'erreur de trajectoire et la dérive d'amplitude | Jalons audio P2 repassés ; pas de P4 avant validation de P3 |
| P4 | Quatre blocs, puis couverture des 24 | Pas de cascade poursuivie avec un préfixe déjà rejeté |
| P5 | Convertir les matrices restantes | Inventaire sans matrice dense oubliée ; fichier réel sous la cible |
| P6 | Qualité et livraison | Test scellé, écoute, durées réelles, ressource et chargement autonome |

Une phase peut produire un résultat scientifique négatif ; cela ne la rend
pas « validée ». Le rapport indique alors le défaut isolé et la prochaine
expérience limitée, sans annoncer un modèle utilisable.

## 5. P0 — corrections bloquantes avant le prochain QAT

### P0.1 Un seul contrat temporel

V7 implémente une fonction partagée qui fabrique le batch t FP32 à partir du
schedule FP32. L'utiliser dans préparation des cibles, train, audit ponctuel,
teacher-on-student, traces et rendu. Vérifier les dtypes par assertions.

Le cache doit inclure les empreintes des poids, buffers, code de forward,
sampler, conditionnement, états, sigmas et leur représentation binaire,
ainsi que versions MLX, dtype de t, crop et durée. Changer la version du
schéma ; **refuser les cibles V6**, même si leurs poids et états sont identiques.

Test réel : même x/conditionnement et les huit sigmas exacts, comparaison
du forward direct, du chemin préparateur de cibles et du chemin trainer.
Avant arrondi éventuel des cibles, les résultats doivent être identiques ;
après stockage FP16, vérifier exactement la conversion attendue.

### P0.2 État reprenable, distinct du fichier d'inférence

Une sauvegarde de train contient :

- Tous les maîtres FP32 encore nécessaires, y compris fenêtres fermées.
- Scales/seuils entraînables et biais vectoriels ; liste exhaustive des
  paramètres entraînables avec noms, shapes et dtypes.
- Moments optimiseur FP32, compteurs, scheduler, RNG MLX/NumPy/Python,
  ordre des données, compteur de microbatch et accumulation si non vide.
- Identité du teacher, scope, quantizer, configuration effective et caches.
- Empreintes et indicateur de complétude écrit après succès de sauvegarde.

Le trainer V7 intègre la reprise du maître, de l'optimiseur et des RNG avec
empreinte de payload ; le test unitaire inter-processus est vert. Le smoke
test réel du modèle reste obligatoire.

Conserver l'état par fenêtres sur disque ; ne charger que la fenêtre
active. Initialiser les nouveaux maîtres depuis **les poids originaux du
teacher**, jamais depuis les codes arrondis d'un export. Réouvrir une fenêtre
depuis son état sauvegardé. Réutiliser training_checkpoint.py après contrôle
de son intégration effective dans le trainer, pas seulement de ses tests isolés.

Test du vrai chemin de train : 20 updates continus contre 10 + sauvegarde +
nouveau processus + 10, avec mêmes exemples/bruits. Comparer maîtres, scales,
bias, moments, pas et codes ; codes identiques et écart FP32 relatif ≤1e-6,
ou expliquer et borner une nondétermination mesurée avant toute longue reprise.
Rejouer aussi fermeture/réouverture d'une fenêtre. Refuser un état incomplet.

### P0.3 QAT dur → sauvegarde → artefact → nouveau processus

Le trainer V7 écrit désormais une fixture du forward dur puis délègue la
relecture à un processus neuf. Cette preuve n'est pas encore exécutée.

Juste avant sérialisation, conserver des tenseurs de contrôle x/t/conditions
et les sorties du **modèle entraîné en forward dur**, pas d'un modèle déjà
reconstruit depuis le teacher.

Comparer ensuite, dans un autre processus :

1. Maîtres → quantizer dur → codes et scales du fichier.
2. Tous les autres paramètres modifiés, notamment les huit biais FFN.
3. Forward QAT dur contre forward packed rechargé, avant tout rendu.

Packing, codes, scales et biais vectoriels : égalité exacte attendue avec
leur représentation exportée. Erreur de sortie liée aux kernels FP16 :
relative L2 ≤1e-3 et cosine ≥0,99999 sur la fixture, avec erreur max absolue
également publiée. Ne pas compenser un écart en ajustant le gain.
Ces tolérances sont des règles d'ingénierie, pas des seuils musicaux.

Le test bias-only +0,125 doit échouer avec l'ancien format et passer avec le
nouveau. Une suppression volontaire du biais, un mauvais dtype de t ou une
altération du code réservé doivent faire échouer les contrôles.

### P0.4 Gradients, quantizer et statuts honnêtes

- Comparer gradients avec/sans recomputation sur poids, scales, biais
  et entrée de fenêtre ; vérifier la propagation par le suffixe gelé.
- Tester forward dur et gradient surrogate avec un oracle indépendant.
  Une différence finie du quantizer dur ne valide pas un STE.
- Évaluer réellement les tableaux MLX avant de lire loss, gradients et mémoire.
- Valider le JSON exécuté ; rejeter champs inconnus et options inopérantes.
  Enregistrer LR réel par update, accumulation, eps, weight decay et seed.
- Séparer software_pass, trajectory_pass, audio_technical_pass,
  musical_review_pending et release_pass. Un gate ponctuel ou un fichier
  qui se recharge ne doit plus suffire à un exit code de livraison réussi.

**Sortie P0 :** rapport de tests + micro-run fermé/repris/exporté.
Pas encore de cascade, ni de test final.

## 6. P1 — données et contrôles appariés

### Références gelées

Teacher : dit_medium_f16.npz, SHA256
f9e5647ea3225818657d47d47ae4b34afa29c0568206ca89566c1a758944a38e.
Enregistrer également runtime, codec, T5, config et environnement installés.

Sur huit pas, schedule actuel, valeur terminale zéro non évaluée par le DiT :

~~~text
1 ; 0.9943755865 ; 0.9844802618 ; 0.9579122663 ;
0.8909031749 ; 0.7455465794 ; 0.5124973655 ; 0.2738850117 ; 0
~~~

Construire ces nombres par le sampler de référence, pas en recopiant les
arrondis décimaux ci-dessus. Sauver bruit initial et bruits de réinjection
explicitement pour les comparaisons appariées.

Contrôles : teacher contre clone dense ; teacher via pipeline réel contre
trace audit ; dernier V6 réévalué avec audit corrigé ; G32 sans QAT sur le
même scope. Ajouter un W4A16 de diagnostic seulement si ces contrôles
laissent un doute sur l'outil : il n'est pas le livrable ternaire.

### Jeux de données

- Garder le train autorisé existant. Identifier œuvres/sessions et détecter
  les doublons PCM/near-duplicates quand les sources sont disponibles.
  Sans accès aux sources, marquer l'indépendance comme non démontrée.
- Validation vocale V6 = développement déjà consulté. Ajouter douze prompts
  instrumentaux de développement définis avant le pilote : percussions,
  basse, piano/accords, textures, mixes denses, styles électroniques/acoustiques.
- Conserver le test 24 prompts fermé. Aucun choix de LR, variante, seuil ou
  arrêt anticipé à partir de lui. Si son indépendance est invalidée par
  l'audit de provenance, remplacer le split avant tout score final.
- Construire d'abord 512 états de train : 16 prompts stratifiés × huit
  sigmas × deux graines × deux sources (trajectoire teacher / latent réel
  bruité). Les conditions des sources restent cohérentes ; ne pas leur
  attribuer des annotations musicales inventées.
- Avec toutes les 85 conditions de train, l'extension équivalente fait
  2 720 états. Elle arrive après le pilote ; ce n'est pas le premier remède.
- Tirage équilibré par prompt, timestep et source ; journaliser la couverture
  unique. « 250 updates » n'est pas « une époque ».

Garder la condition 12 s pour les crops courts ; durée décodée actuelle de
128 latents : 11,8886 s. Tester les durées longues avec la longueur latente
calculée par le runtime, pas uniquement une modification de --seconds.

## 7. P2 — faire apprendre réellement le petit modèle ternaire

### Mesures V7 déjà obtenues

Les chiffres pré-v3 de cette section sont des résultats historiques, issus
des caches/validations alors en vigueur. Ils restent utiles pour retracer les
choix, mais la décision actuelle repose sur la répétition avec cache v3 et RNG
corrigé, documentée plus bas et dans le rapport d'exécution.

Le cache d'états retenu contient 512 entrées FP32 source, équilibrées sur
16 prompts, huit sigmas du sampler et deux sources (latent bruité / rollout
teacher), avec deux graines par combinaison. Les targets teacher sont
présentes pour 512/512 entrées ; cache et teacher/T5 concordent avec leurs
empreintes déclarées. Pic Metal de préparation observé : 3,66 Gio.
Le premier cache, trop concentré sur des sous-genres électroniques, est
conservé comme trace mais exclu de l'entraînement ; `balanced-v2` reste
historique et le cache v3 corrigé est désormais la source des nouveaux runs.

**Correction P0/P1 après audit du sampler :** `balanced-v2` avait initialisé
les ré-injections à partir du même flux PRNG que le bruit initial, contrairement
au runtime (`seed` initial, `seed+1` sampler). Le helper partagé et un test
contre `sample_flow_pingpong` corrigent cette parité. Son empreinte a changé :
`balanced-v2` et ses targets restent des preuves historiques mais ne peuvent
plus alimenter un nouveau run. Le cache et les targets v3 ont été construits ;
le P2 répété et ses rendus bruts sont documentés en §14.

Un contrôle **sans QAT**, hard G32 symmetric sur blocs 0–1, sert désormais
de baseline appariée. Sur la validation historique (12 prompts × 8 sigmas),
cosinus velocity moyen = **0,8218**, minimum = **0,4895**, erreur relative
L2 moyenne = **0,6031**. Le cosinus terminal d'état moyen = **0,4252**,
minimum = **0,2331**. Le gate d'audit a donc échoué ; ceci est le zéro de
référence et non un candidat livrable. Le reload inter-processus de cette
baseline reproduit exactement le forward sur trois fixtures (erreur max 0,
cosinus 1,0), ce qui isole ici la qualité du modèle de la fidélité du format.

Le micro-overfit A (blocs 0–1, G32 symmetric, 16 états fixes, accumulation 4,
LR 1e-5) a terminé 50 updates avec checkpoint toutes les 25, records-only et
limite Metal 11 Go. La loss d'entraînement passe de 0,261228 à 0,038251
(−85,4 %). À update 50, 428 169 codes diffèrent du départ,
soit 0,3781 % ; 163 064 codes ont changé depuis l'audit précédent. Tous les
16 états ont été vus, gradients finis, aucun clipping à l'update 50 et pic
Metal = 6,01 Gio. Le roundtrip inter-processus passe sur les trois fixtures
(relative L2 max = 0 ; cosinus min = 0,99999994 ; erreur absolue max = 0).
Ces preuves établissent que le mécanisme optimise et se recharge, **pas** qu'il
améliore à lui seul la qualité musicale.

L'audit apparié à la baseline sur les **12 prompts vocaux historiques**
montre un gain réel : cosinus velocity moyen **0,9065** contre 0,8218 ;
erreur relative velocity moyenne **0,4154** contre 0,6031. En fin de
trajectoire, le cosinus d'état moyen vaut **0,5490** contre 0,4252 ; la moyenne
des carrés d'erreur d'état relative baisse de **3,0598 à 1,3936 (−54,5 %)**.
Le pire carré d'erreur passe de 5,5688 à 2,1305 (pas de régression du pire
cas). Le critère numérique P2 de −20 % passe donc sur ce split.

Cela **ne valide pas P2** : le pire cosinus velocity reste **0,6290**
(prompt chant, 136 BPM, sigma 0,9579), sous le gate candidat 0,80 ; l'auditeur
rapporte `candidate_velocity_pass=false`. Ce premier résultat vocal a donc
déclenché la création d'un second développement instrumental prompt-disjoint.

Le challenger LR 1e-4 apparié (même teacher, cache, 16 états, seed 20260923,
50 updates) a lui aussi passé le roundtrip et fait davantage bouger les codes :
loss finale 0,04236 et 2,624 % de codes différents du départ, contre 0,03825
et 0,378 % pour LR 1e-5. Sur les 12 prompts vocaux, il est moins bon
(cosine velocity moyen **0,8453**, terminal moyen **0,4665**, NMSE terminale
moyenne **1,5142**, pire **3,4231**) ; sur l'instrumental, sa NMSE terminale
moyenne est meilleure, mais ses cosinus velocity sont inférieurs. Cela
confirme un compromis, pas une victoire par nombre de flips ; les deux LR
ratent encore le minimum velocity.

Pas de cascade vers quatre blocs. Un développement instrumental
prompt-disjoint a été composé dans
[`instrumental-dev-selection.json`](../output/sample-expertise-pilot/ternary-quality-v7-20260924/instrumental-dev-selection.json)
à partir du corpus local autorisé : 12 prompts/sources inédits pour le QAT,
hashes latents/métadonnées vérifiés, aucun chevauchement avec les rôles
train/validation/test antérieurs. Les 12 prompts couvrent des styles
électroniques/dub de ce corpus ; cela améliore le contrôle mais ne remplace
pas une validation acoustique large. Sur cette validation, LR 1e-5 atteint
un cosinus velocity moyen **0,9405** (baseline 0,8675), minimum **0,7883**
(baseline 0,6129), cosinus terminal moyen **0,8344** (baseline 0,6922),
NMSE terminale moyenne **0,6630** contre 4,2669 (−84,5 %) et pire **1,4272**
contre 8,8371 (−83,9 %). Gros progrès apparié ; le gate velocity reste
techniquement rouge d'un cheveu (minimum 0,7883 < 0,80), ce qui interdit
encore P2. Le LR 1e-4 baisse davantage la NMSE terminale instrumentale
(0,4283 contre 0,6630), mais baisse aussi le cosinus velocity moyen
(0,9274 contre 0,9405) et son minimum (0,6875 contre 0,7883) ; le cosinus
terminal minimum est 0,7391 contre 0,7428. Aucun ne domine sur tous les
critères et les deux ratent le gate minimum velocity. La seconde seed LR 1e-5
confirme le résultat instrumental : cosine velocity moyen **0,9399**, minimum
**0,7926**, cosine terminal moyen **0,8381**, NMSE terminale moyenne **0,6860**
et pire **1,5300** ; les gains sur baseline restent ~84 % en moyenne et ~83 %
au pire. Le minimum reste juste sous 0,80, de façon reproductible. Sur les
12 prompts vocaux, la seed 2 donne cosine velocity moyen **0,9116**, minimum
**0,6922** (même cas chant, 136 BPM, sigma 0,9579), cosine terminal moyen
**0,5698**, NMSE terminale moyenne **1,3107**, pire **2,0650** ; seed 1 :
0,9065 / 0,6290 / 0,5490 / 1,3936 / 2,1305. La confirmation est cohérente
et améliore les moyennes/pire cas, mais `candidate_velocity_pass` reste faux
sur ce prompt de chant. Le test réservé est fermé. L'export partiel 0–1 de la
seed 2 a été rendu dans un processus neuf (2 505 657 236 octets, paramètres
rechargés exactement ; ce fichier reste explicitement non livrable). Sur
trois générations brutes 12 s, sampler 8 pas, les trois paires sont finies,
stéréo, de durée 11,8886 s et gardent une corrélation spectrale/enveloppe
~0,92, mais `technical_pass=false` : RMS student/teacher = **1,885 / 1,954 /
1,667** ; peak ratio = **1,884 / 1,621 / 1,534**. Les pics étudiants sont
1,19–1,46 (>1,0 float). Aucune normalisation/limitation appliquée. Pic Metal
audio = **8,60 Gio**, sous la garde 11 Go.

Le terminal latent a déjà un RMS ratio de **1,30–1,39** et un écart-type
student/teacher de **1,38–1,46** sur ces mêmes exemples ; l'amplification
audio s'accentue au décodage. Hypothèse à tester, pas causalité prouvée :
la loss velocity ponctuelle n'empêche pas la dérive d'amplitude qui s'accumule
sur huit pas. P2 échoue donc malgré ses NMSE moyennes ; pas de conversion de
blocs supplémentaires. La prochaine exécution doit ablater P3 (loss de
trajectoire différentiable) contre A, suivre le RMS latent à chaque sigma,
réauditer le rollout complet huit pas et repasser les mêmes gates audio bruts.

### Expérience A : contrôle corrigé, minimal

Fenêtre 0–1 ; teacher dense gelé ailleurs ; G32 symmetric actuel ; maîtres
originaux FP32 persistants ; forward **dur** ; loss sur la sortie velocity
complète après le suffixe, pas seulement sur les activations du bloc.

Avant 250 updates : micro-overfit sur 16 états fixes, jusqu'à 50 updates.
Mesurer la baisse de loss dure, la conservation des biais et les changements
réels de codes/scales. Une loss douce seule ne compte pas.

Réglages initiaux explicites : AdamW FP32, weight decay 0, eps=1e-6,
microbatch 1, accumulation 4, clip norme globale 1. Deux LR maîtres au
maximum pour le smoke : 1e-5 et 1e-4, 50 updates chacun, même initialisation.
Choisir sur stabilité et baisse de loss dure ; ne pas recycler le mauvais
état du premier essai. Ce sont des points de départ à mesurer, pas une
recette publiée pour SA3. Sauver le scheduler avec l'état.

Budget pilote : 250 updates ; audit léger et checkpoint toutes les 25.
Extension à 500 puis 1 000 maximum seulement si la validation progresse
encore. Choisir le meilleur checkpoint de développement, pas forcément le
dernier. Arrêt anticipé après trois audits consécutifs sans amélioration
relative de 1 % de l'erreur terminale moyenne ; conserver les mesures du
pire cas pour ne pas améliorer la moyenne en sacrifiant un prompt.

Loss de départ, moyenne par exemple puis moyenne du batch :

~~~text
NMSE(vS, vT) = mean((vS-vT)^2) / max(mean(vT^2), epsilon_train)
Lv = NMSE(vS, vT) + 0.1 * (1 - cosine(vS, vT))
~~~

Fixer epsilon_train avant entraînement, d'après les énergies du train,
enregistrer sa valeur ; reductions FP32. Réévaluer la loss dure et les
métriques en FP32 indépendamment de la valeur utilisée pour backward.

Journal obligatoire, par couche :

- Normes et ratios update/poids ; gradients non finis ; clips.
- Occupation {-1,0,+1}, transitions depuis l'update précédente ET le départ.
- Distance normalisée aux seuils, quantiles des scales, groupes nuls.
- Loss et erreurs velocity/débruitage par timestep et famille de prompt.
- Mémoire active/pic, RSS/swap, temps/update et exemples uniques vus.

Pas de quota artificiel de flips : changer des codes n'est pas une qualité.
Mais perte élevée + codes durablement immobiles interdit de prétendre
qu'une réorganisation ternaire a été apprise.

### Expérience B : un seul challenger si A plafonne

Quantizer symétrique à scale apprise, inspiré de LSQ :

~~~text
s = exp(log_s)
s_eff = représentation FP16 de s, avec chemin de gradient explicite
u = W_master_fp32 / s_eff
q_hard = clip(round(u), -1, +1)
W_forward = FP16(s_eff * q_hard)
~~~

Initialiser s par une minimisation scalaire de l'erreur du groupe teacher,
déterministe et testée ; conserver W_master non arrondi. Seuil fixé à s/2,
pas de seuil libre supplémentaire pour cette première comparaison.
Gradient surrogate de poids et scale spécifié et testé, avec facteur
1/sqrt(N_g) pour le pas d'un groupe de N_g valeurs utiles ; l'adaptation
groupée n'est pas une reproduction établie de LSQ sur SA3.

Optimiseur séparé pour log_s, LR initial 1e-3, weight decay 0, états FP32 ;
publier son effet réel. Mêmes données, seeds, scope et 250 updates que A.
Tout groupe dont le scale devient nul/non fini après cast FP16 invalide
l'update ; ne pas le laisser disparaître silencieusement.

Le forward reste dur, donc exportable à chaque jalon. Ne pas ajouter en
même temps Hadamard, seuils libres, corpus supplémentaire et loss de rollout.
CAT-Q ou un véritable calendrier soft-to-hard constituent une recherche
ultérieure distincte si A et B échouent ; ne pas baptiser le surrogate
V6 à sharpness fixe « reproduction CAT-Q ».

### Passage P2

Comparer à l'initialisation dure de **même scope**, audit P1 corrigé :

- Contrats P0 toujours verts après train et reload.
- Baisse d'au moins 20 % de NMSE terminale moyenne sur développement ;
  pire NMSE ne se dégrade pas de plus de 5 %.
- Cosinus velocity moyen ≥0,90 et minimum ≥0,80 sur le développement ;
  un bon score moyen ne masque pas un prompt catastrophique.
- Trois paires audio brutes décodées, pas de silence, souffle dominant,
  écrêtage nouveau ou rupture grossière ; RMS student/teacher dans
  [0,70, 1,30], peak ratio dans [0,50, 1,50], écoute nécessaire.
- Confirmer le finaliste sur une seconde seed d'entraînement, mêmes graines
  d'évaluation. Les deux résultats et leurs écarts restent publiés.

Ces valeurs sont des gates de progression fixés pour V7, pas des lois
perceptuelles. Si A passe tous les gates, il est permis de continuer avec A
sans complexifier. Si A/B apprennent sur train mais échouent en trajectoire
ou sur l'audio brut, passer à P3 seulement comme ablation bornée, sans ouvrir P4.
Si même le micro-overfit ne progresse pas, retour P0/quantizer : ne pas
convertir plus de blocs pour masquer le problème.

### Résultat P2 répété sur cache v3

P2 blocs 0–1, G32 symmetric, LR `1e-5 → 1e-6`, 50 updates, 16 états,
seed `20260924` : loss `0,20842 → 0,09351`, 331 416 codes changés
(0,293 %), pic Metal 6,010 Go, round-trip exact. Sur les mêmes 12 prompts de
développement avec RNG corrigé : cosine velocity **0,93262 / 0,78394**
(moyenne/minimum, minimum requis 0,80), cosine terminal **0,82143**,
RMS latent terminal **1,534**. Rendu brut trois prompts : RMS
**2,002 / 1,593 / 1,796**, peak **1,937 / 1,770 / 1,633** ; échec 3/3.
Erreur de sampler v2 corrigée n'était donc pas cause unique. Aucun passage
à P4.

## 8. P3 — corriger l'erreur de trajectoire et la dérive d'amplitude

Ne pas commencer par « deux pas tardifs uniquement ». Le premier pas
sigma=1 est déjà mauvais sur une sonde V6 et le bruit commun masque une
partie des erreurs intermédiaires.

**P3 est maintenant obligatoire avant P4** : P2 apprend et s'améliore en
latents, mais le rendu brut amplifie RMS ×1,67–1,95 et les pics dépassent
1,0. Comparer au contrôle A avec le même scope, états, seeds et budget ;
aucun gain correctif/post-traitement dans l'évaluation.

1. Capturer des états du student courant sur les prompts **train**.
   Sur chacun, calculer la cible teacher avec exactement les mêmes x, t,
   conditions. Référencer le hash du student producteur.
2. Pour la moitié des segments, choisir une paire de pas consécutifs de la
   grille huit pas ; équilibrer les sept paires, y compris la première et
   la paire qui mène au terminal. Garder l'autre moitié en loss ponctuelle.
3. Student et teacher partent du même état ancre et utilisent les mêmes
   bruits de réinjection. Le teacher suit sa propre branche depuis l'ancre ;
   ses états cibles à un/deux pas sont précalculés. Les seconds états
   student et teacher peuvent alors différer : c'est l'objet de cette loss.
4. Différencier la branche student sur deux pas, avec recomputation.
   Contrôle : Lv seul, même état de départ et même budget.

Objectif proposé : Lv + 0,25*Ltraj, Ltraj = NMSE des états après deux pas,
en incluant le terminal lorsque la paire l'atteint. Coefficient figé pour
cette ablation ; ne pas prétendre que sa valeur est théoriquement optimale.
À chaque audit, publier en plus le RMS ratio des états à chacun des huit
sigmas, le ratio latent terminal, puis RMS/peak bruts décodés. Le test porte
sur le rollout complet à huit pas : un gain local à deux pas ne suffit pas.
Une perte sur x−sigma*v au **même x** est surtout une repondération de
velocity par sigma : elle n'ajoute pas seule la supervision de trajectoire.

Cache figé pendant une expérience ; nouveau cache pour une nouvelle passe,
avec manifeste et budget comparable au contrôle. Teacher libéré avant le
backward student. Mesurer le pic réel à deux pas avant 250 updates et garder
le garde-fou Metal 11 Go sur préparation, entraînement et rendu.

Retenir la variante seulement si elle améliore la NMSE terminale moyenne
d'au moins 10 % contre son contrôle apparié, sans régression du pire cas
>5 %, passe le gate velocity candidat (moyenne ≥0,90, minimum ≥0,80) et
fait réussir les trois gates audio bruts, dont RMS [0,70;1,30] et peak
[0,50;1,50]. Pas de normalisation. Sinon conserver le contrôle, documenter
le résultat et ne pas multiplier des variantes non isolées.

### Résultat de l'ablation P3 du 24/09/2026

Cache : 224 paires (16 prompts train × 2 graines × 7 transitions), 32 par
transition ; teacher calculé après libération de l'élève. Contrôle A (velocity
seule) et P3 ont partagé source records, ancres, targets, seeds et 50 updates.
Pic Metal : 6,435 Go (A), 6,917 Go (P3). Les deux round-trips sont exacts.

P3 : cosine velocity moyen/minimum **0,93633/0,79247** ; terminal moyen
**0,83519** ; NMSE terminale **0,73714** contre **0,74777** pour A, soit
seulement **−1,42 %** au lieu des −10 % requis. RMS terminal **1,469**.
Les deux variantes ont **zéro code dur changé**. Le rendu brut reste hors
limites sur chaque prompt : P3 RMS **1,932/1,538/1,700**, peak
**1,874/1,798/1,525** ; contrôle A RMS **1,939/1,546/1,708**, peak
**1,880/1,803/1,533**. Gains audio <1 %, sans normalisation.

**Décision : P3 échoue ; P4 fermé.** Les reprises A/P3 partaient du checkpoint
ternaire arrondi et reconstruisaient un nouveau maître FP32 par déquantification.
Le rapport d'exécution propose un seul essai apparié P3.1 qui conserve le
maître FP32 P2 ; aucune exécution de cet essai ni extension de scope sans
autorisation.

### P3.1 proposé — reprise fidèle, couverture complète, pas d'élargissement

Le checkpoint P2
[`window_latest.npz`](../output/sample-expertise-pilot/ternary-quality-v7-20260924/micro-overfit-0-1-g32-symmetric-lr1e-5-seed20260924-v3-runtime-rng-fix1/checkpoints/window_latest.npz)
existe (1,2 Go) avec sidecar v4 vérifiable. Il contient les maîtres FP32,
moments Adam FP32, étape 50, LR `1,0089e-6` et état RNG. La reprise stricte
actuelle valide `run_signature` exact ; changer loss/cache P2→P3 doit donc
passer par une nouvelle voie explicite `warm-start`, jamais en désactivant ce
contrôle.

1. Implémenter le warm-start de provenance : vérifier sidecar, hash, scope,
   formes et dtypes ; charger maîtres et moments, noter le checkpoint parent,
   cache, seed et nouvelle signature d'expérience. Ne modifier ni le fichier
   source ni le chemin de reprise exacte.
2. Préflight sans update : recalculer les codes durs depuis les maîtres et
   comparer aux records P2 ; recharger le forward dur et comparer les fixtures
   P2 ; restaurer moments/compteur ; vérifier par smoke le LR effectif. Tout
   écart hors tolérance bloque l'entraînement.
3. Depuis deux lectures propres du même checkpoint immuable, comparer A
   (velocity seule) et P3 (`Lv + 0,25 × NMSE_endpoint`). Mêmes maîtres,
   moments, ordre RNG, cache, hyperparamètres et coût ; seul le terme de
   trajectoire change. LR fixe égal au taux vérifié du checkpoint, journalisé
   pour chaque update.
4. Utiliser le cache de 224 paires sans remise. Avec 2 microbatches trajectoire
   par update, faire 112 updates pour couvrir toutes les transitions une fois ;
   A suit le même ordre d'ancres et calcule la loss ponctuelle, y compris sur
   les ancres où P3 fait l'unroll. Publier couverture, flips persistants,
   distances aux seuils, mouvement des maîtres et scales à mi-parcours puis à
   la fin.
5. Garder tous les gates P3 existants. Si préflight, apprentissage des codes,
   fidélité velocity/trajectoire ou audio brut échoue, fermer P4 et diagnostiquer
   le quantizer/l'objectif ; ne pas convertir davantage de blocs. Si tout passe,
   répéter une seconde seed avant d'autoriser 0–3.

P3.1 n'est pas lancé. La modification du contrat de reprise et l'expérience
requièrent une autorisation de lancement distincte ; ce plan n'est pas une
promesse que la qualité sera atteinte.

## 9. P4 — quatre blocs d'abord, 24 ensuite, avec retours sur les précédents

Le point décisif est **0–3**, là où V6 s'est arrêtée.

- Ajouter 2–3 depuis le teacher, tout en conservant les maîtres/moments de
  0–1 ; faire un passage 2–3 puis rouvrir 0–1 avec son vrai état.
- Si la mémoire le permet, comparer une courte récupération conjointe 0–3 ;
  sinon alterner les paires. Le gradient porte toujours sur la sortie complète.
- Auditer le contrôle sans QAT et le candidat sur le même scope 0–3.
  Garder les gates P2, puis exiger un cosinus terminal moyen ≥0,90 et
  minimum ≥0,80 sur le développement figé, plus les écoutes pilotes.
  Ces seuils sont une barrière d'ingénierie annoncée avant le run, pas une
  équivalence entre cosinus et musicalité.
- Si ce jalon manque après le budget A/B/P3 borné, **pas de cascade 24 blocs**.
  Produire une conclusion sur l'hypothèse encore en défaut et chiffrer une
  nouvelle expérience ; pas de passage automatique à un calcul payant.

Une fois 0–3 accepté : ajouter les paires 4–5, …, 22–23. À chaque extension,
entraîner la paire nouvelle et revisiter les fenêtres déjà quantifiées
avec leurs états persistants. C'est une optimisation par coordonnées du
modèle complet, pas une succession de blocs figés définitivement.

Jalons de validation complète : 4, 8, 16, 24 blocs. Entre eux, audit léger
par paire et arrêt en cas de régression brutale. Rafraîchir les états
student de train aux jalons, pas ceux de validation.

Budget initial indicatif par paire : 500 updates, puis une passe de
récupération de 250 ; tout dépassement requiert une courbe de validation
encore favorable. Recalculer l'ETA après le pilote. Ne pas promettre que
24 fois une reconstruction locale suffira.

Toujours sélectionner et reprendre un checkpoint **dur** complet. Des
fichiers intermédiaires d'inférence peuvent dépasser 500 Mo ; ils portent
explicitement le statut partial_scope et ne sont pas livrés.

## 10. P5 — couvrir réellement tout le modèle et respecter la taille

Après validation du core, inventorier par nom chaque matrice restante.
Garder la liste d'exceptions vectorielles fermée ; échouer sur toute matrice
dense non déclarée. Les adaptateurs de chargement doivent couvrir :

- Conditionnement local, dont dimension d'entrée 257 et padding G32.
- Projections globales, temporelles, gates matriciels et durée.
- Entrées/sorties, convolutions et tokens mémoire appris.

Tester chaque chemin avec des conditions **non nulles** représentatives,
pas seulement le text-to-audio dont certaines branches restent inactives.
Pour les tokens mémoire et embeddings, tester la lecture effective ;
le modèle ne se résume pas à remplacer nn.Linear.

Convertir par famille, garder les autres poids gelés, puis revisiter les
fenêtres sensibles. Mesurer d'abord la sensibilité ; ne pas modifier
plusieurs familles simultanément si l'audio se dégrade.

Réévaluer P0/P4 après chaque famille. Une petite matrice FP16 laissée parce
qu'elle est sensible signifie **scope incomplet**, pas « 100 % ternaire ».
Un hybride peut servir de diagnostic, mais changer le livrable final
demande une décision explicite.

Export autonome du DiT : configuration, matrices packed, vecteurs/buffers,
manifeste et empreintes. Vérifier le chargement sans accès au teacher
original : aucun fallback silencieux ne doit restaurer une matrice dense.
T5 et SAME-L restent des dépendances explicites hors DiT.

## 11. P6 — acceptation réelle et écoute

### Avant d'ouvrir le test

Figer checkpoint, hashes, code, sampler, graines, prompts, durées,
coefficients et critères. L'acceptation n'est pas un booléen produit par
le seul auditeur ponctuel.

Développement : 24 prompts (12 validation historique + 12 instrumentaux),
deux graines par prompt, rendus courts puis vrais 30 s ; quelques
générations à la durée longue réellement visée par le produit.
Profiler les durées longues séparément : le pic à 128 latents n'est pas
une preuve pour 180 s.

### Dossier d'acceptation

- **Logiciel :** parité QAT dur/export/reload, reprise fidèle, provenance,
  contrat ternaire et scope exhaustif, absence de dépendance au teacher.
- **Trajectoire :** erreur relative, cosinus et RMS ratio par sigma,
  prédiction débruitée et latent terminal ; moyenne, médiane, pire prompt.
  Conserver la barrière P4 sur développement ; ne pas transformer des états
  dominés par le bruit en preuve de qualité.
- **Audio brut :** FLOAT WAV, stéréo, sample rate et durée vérifiés,
  NaN/Inf, silence, DC, sample/true peak, RMS/LUFS et artefacts temporels.
  Aucun EQ, limiteur, normaliseur ou post-traitement pour faire passer le run.
  Mesures spectrales/enveloppes publiées comme diagnostics, pas comme
  score universel de musicalité.
- **Écoute aveugle :** ordre A/B masqué, identité révélée après notation ;
  naturel/timbre, artefacts, structure et respect du prompt. Le niveau brut
  reste archivé ; une copie à niveau égal peut aider l'écoute, identifiée
  comme dérivé sans remplacer les mesures brutes.
- **Règle musicale proposée :** au moins 80 % des paires jugées utilisables
  sans dégradation importante, écart médian student−teacher ≥−0,5 sur
  une note de qualité à cinq niveaux ; aucun défaut grave reproductible
  (silence, souffle dominant, saturation ou instabilité récurrente).
  Guillaume valide l'écoute ; sans cette revue, musical_review_pending.
  Ce petit panel n'est pas une estimation universelle de qualité.
- **Ressources :** octets du DiT, paquet complet, pic accélérateur,
  RSS/swap, latence et durée effectivement générée.

Ouvrir les 24 prompts test réservés seulement après ce gel. Exécuter les
mêmes vérifications avec au moins deux graines ; les familles couvertes
et absentes sont indiquées. Si l'on modifie le modèle après les résultats,
ce test devient du développement et un nouveau test est nécessaire.

Ne déclarer release_pass qu'avec toutes les preuves réunies. Une écoute
peut être bonne alors qu'un indicateur de fidélité est insuffisant :
documenter le désaccord et décider avant une nouvelle évaluation, ne pas
abaisser discrètement les seuils après observation.

## 12. Budget local et protection des artefacts

- Limite accélérateur : **12 000 000 000 octets** (~11,18 GiB).
  Garde préventive : 11 000 000 000 octets (~10,24 GiB).
- Machine observée : 24 GiB unifiés. Publier RSS/swap en plus de Metal ;
  mémoire du modèle et mémoire du processus ne sont pas la même métrique.
- V6 mesurée : deux blocs avec cibles offline ~6,02 GiB ; quatre blocs
  ~8,74 GiB. Ce ne sont pas des garanties pour LSQ, deux pas ou 30 s.
- Teacher seul pour les cibles, puis libération avant optimisation.
  Une fenêtre de 1–2 blocs active ; quatre seulement après mesure.
  Profil de 20 updates avant chaque changement de scope/durée.
- FP32 maître + deux moments = ~17,43 Go pour toutes les matrices sur disque.
  Avec gradients simultanés, ~23,24 Go avant activations : pas de QAT
  globale naïve en mémoire sur cette machine.
- Garder les états de fenêtres fermées sur disque, deux versions de la
  fenêtre active et un meilleur export packed. Ne pas copier un teacher
  de 2,9 Go dans chaque répertoire ni conserver tous les exports partiels.
- Interne : ~51 GiB libres lors de l'audit. Externe : ~7,9 GiB seulement.
  Préflight disque obligatoire ; prévoir ~30 GiB de travail additionnel
  et garder ≥15 GiB libres. Ne pas déplacer de gros caches vers l'externe
  sur l'hypothèse qu'il a de la place.
- Les archives historiques ne sont jamais supprimées automatiquement.
  Si la marge manque, demander quels artefacts jetables peuvent être
  déplacés/supprimés ; ne pas détruire les preuves ni les états reprenables.

Les raffinements V6 de quatre blocs prenaient environ 30 min pour 250
updates. Un programme de milliers d'updates représente donc potentiellement
des dizaines d'heures, pas une relance symbolique. Budgets exacts et ETA
viennent du pilote ; les règles d'arrêt empêchent d'y consacrer tout ce
temps sans signal positif.

Aucun cloud payant, allocation GPU ou changement d'infrastructure implicite.
Si deux formulations correctement testées plafonnent sous la contrainte
locale, présenter les compromis calcul/qualité/scope, sans renommer l'échec.

## 13. Fichiers à modifier, dans cet ordre

Éviter une nouvelle pile de scripts parallèles. Étendre les composants
existants avec un schéma V7 explicite et des tests de régression.

| Composant existant | Travail V7 |
| --- | --- |
| [prepare_ternary_teacher_targets.py](../services/musicgen/prepare_ternary_teacher_targets.py), [ternary_teacher_targets.py](../services/musicgen/ternary_teacher_targets.py), [build_ternary_state_cache.py](../services/musicgen/build_ternary_state_cache.py) | Dtype temporel partagé, empreintes teacher/T5/code/données, 16 prompts stratifiés, huit sigmas du sampler, mémoire plafonnée |
| [training_checkpoint.py](../services/musicgen/training_checkpoint.py), [train_ternary_window_v6.py](../services/musicgen/train_ternary_window_v6.py), [verify_ternary_records_roundtrip.py](../services/musicgen/verify_ternary_records_roundtrip.py), [build_ternary_hard_baseline.py](../services/musicgen/build_ternary_hard_baseline.py) | Reprise inter-processus, contrôle gradient/code/scale/mémoire, baseline hard appariée et golden forward rechargé dans un processus neuf |
| [train_ternary_quality.py](../services/musicgen/train_ternary_quality.py), [ternary_contract.py](../services/musicgen/ternary_contract.py), [export_ternary_records.py](../services/musicgen/export_ternary_records.py) | Biais vectoriels, golden forward avant save, quantizer dur et challenger scale appris |
| [audit_ternary_quality.py](../services/musicgen/audit_ternary_quality.py) | Grille runtime, teacher/student au même t, terminal, contrôles appariés, vrais statuts |
| [render_ternary_quality.py](../services/musicgen/render_ternary_quality.py) | Durée réellement générée, métriques brutes, split manifest vérifié, garde Metal 11 Go, séparation technique/écoute |
| [Tests dans services/musicgen/tests](../services/musicgen/tests) | Régressions temps/biais/reprise/échantillonnage/gradients ; apprentissage réel et scope intégral restent à exécuter |

Les flags et commandes du futur trainer ne sont pas inventés dans ce plan :
ils doivent être implémentés et testés à P0. Les anciennes commandes V6
restent celles de l'expérience archivée, pas un moyen d'exécuter V7.

## 14. État actuel et toute prochaine action

Exécution V7 désormais complète jusqu'au gate d'arrêt P3. Le pipeline
cache/RNG, checkpoint, export et rendu fonctionne sous la limite locale ;
**aucun candidat ne passe la qualité audio**.

- **P2 v3 répété :** velocity mean/min **0,93262/0,78394**, terminal cosine
  **0,82143**, RMS terminal **1,534** ; audio RMS **2,002/1,593/1,796**,
  peak **1,937/1,770/1,633**. P2 refusé.
- **P3 apparié :** gain NMSE terminale **1,42 %** contre seuil 10 % ; velocity
  minimum **0,79247**, terminal cosine **0,83519**, RMS latent terminal
  **1,469**. Aucun code dur n'a changé durant P3 ; audio RMS
  **1,932/1,538/1,700**, peak **1,874/1,798/1,525**. P3 refusé.
- **Ressources :** pic Metal maximum 6,917 Go entraînement et 8,604 Go rendu,
  sous le garde de 11 Go décimaux. Tous les round-trips exportés sont exacts.
- **Décision :** P4 fermé ; aucun scope 0–3, aucune conversion des 24 blocs,
  aucun test réservé ouvert. Aucun artefact final ≤500 000 000 octets ni
  écoute humaine formelle.
- **Cause désormais établie :** le warm-start exact P3.1 ne suffit pas. Le
  symmetric G32 et la loss de trajectoire restent trop rigides; prolonger à LR
  fixe dérive après P2 et ne change que ~0,06 % des codes. Le nouveau plan
  impose un quantizer TTQ/LSQ, calibration comportementale, transition
  progressive et sélection validation avant toute cascade.

Artefacts : [cache v3](../output/sample-expertise-pilot/ternary-quality-v7-20260924/state-cache-512-fp32-balanced-v3-runtime-rng-fix1/manifest.json),
[audit P2](../output/sample-expertise-pilot/ternary-quality-v7-20260924/micro-overfit-0-1-g32-symmetric-lr1e-5-seed20260924-v3-runtime-rng-fix1/audit-instrumental-dev-runtime-rng-v3/audit_summary.json),
[audio P2](../output/sample-expertise-pilot/ternary-quality-v7-20260924/micro-overfit-0-1-g32-symmetric-lr1e-5-seed20260924-v3-runtime-rng-fix1/audio-pilot-3prompt-8steps-v3/audio_metrics.json),
[audit contrôle A](../output/sample-expertise-pilot/ternary-quality-v7-20260924/control-a-pointwise-pairanchors-0-1-g32-50updates-seed20260924-fix1/audit-instrumental-dev-paired-A/audit_summary.json),
[audio contrôle A](../output/sample-expertise-pilot/ternary-quality-v7-20260924/control-a-pointwise-pairanchors-0-1-g32-50updates-seed20260924-fix1/audio-pilot-3prompt-paired-A/audio_metrics.json),
[audit P3](../output/sample-expertise-pilot/ternary-quality-v7-20260924/p3-trajectory-weight0p25-pairanchors-0-1-g32-50updates-seed20260924-fix1/audit-instrumental-dev-paired-P3/audit_summary.json),
[audio P3](../output/sample-expertise-pilot/ternary-quality-v7-20260924/p3-trajectory-weight0p25-pairanchors-0-1-g32-50updates-seed20260924-fix1/audio-pilot-3prompt-paired-P3/audio_metrics.json).

## 15. Addendum P3.1

P3.1 a été exécuté après préflight : 224/224 paires, maîtres FP32 et moments
Adam restaurés, round-trip passé. Sur le corpus exact du cache, Contrôle A est
à `0,92804/0,73204` (mean/min) et terminal `0,75743`; P3.1 à
`0,92819/0,73154` et terminal `0,75873`. L'audio brut échoue dans les deux
cas. Le premier audit sur `universal-dataset/latents-12s` est exclu car il ne
correspondait pas au corpus du cache.

Un pilote learned-symmetric lancé depuis les records ternaires reste rouge
(`0,91795/0,71337`, terminal `0,70638`). Il n'est pas une preuve contre un
quantizer appris lancé depuis le teacher dense, mais il confirme qu'une courte
reprise sur une projection déjà perdue n'est pas une réparation. Voir le
[plan V7 révisé](TERNARY_QUALITY_RECOVERY_PLAN_V7.md) et l'[addendum du rapport](TERNARY_V7_EXECUTION_REPORT_2026-09-24.md).
