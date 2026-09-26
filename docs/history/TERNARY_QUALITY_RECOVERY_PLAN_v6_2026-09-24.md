# Plan v6 — réussir la ternarisation de Stable Audio 3 Medium

Date : 23 septembre 2026. Statut : pilotes correctifs exécutés, cascade en cours ; modèle complet non acquis.
Ce document remplace les plans v3 et v5. [V3 conservé intégralement](archive/TERNARY_QUALITY_RECOVERY_PLAN_v3_2026-09-22.md).

## 1. Décision

**Convertir le modèle préentraîné existant. Ne pas repartir de zéro.**

Voie principale : quantification symétrique avec paramètres de quantification
appris, transition douce puis codes durs, optimisation de la sortie complète
par fenêtres réouvertes, distillation sur états réellement rencontrés, puis
conversion de toutes les matrices. G32 redevient une option sérieuse.

Trois différences avec les essais rejetés :

1. Apprendre la quantification ; ne plus seulement appliquer absmean + STE.
2. Optimiser la fonction finale et revisiter les blocs ; ne plus figer chacun
   définitivement après 200 mises à jour.
3. Financer des groupes plus précis en convertissant aussi les matrices encore
   FP16. Ne pas confondre taille d'un jalon partiel et taille finale.

Hypothèse de travail : les échecs viennent au moins en partie de la méthode
d'optimisation, de sa couverture et du protocole. Ils ne démontrent pas une
impossibilité intrinsèque du ternaire. Réussite possible, non garantie.

## 2. Diagnostic corrigé

Faits vérifiés dans le code et les rapports existants :

- **V4 échoue réellement aux contrôles employés.** Export de 479 660 087 octets,
  reload exact ; cosinus velocity moyen/minimum 0,81248/0,65934 sur validation,
  0,80827/0,69551 sur l'ancien test. Trois rendus courts hors critères audio.
  Cela sépare bien conformité du fichier et qualité de la fonction.
- **L'objectif reste surtout local.** Dans
  [train_ternary_quality.py](../services/musicgen/train_ternary_quality.py),
  seuls les poids du bloc courant sont entraînés ; seul le bloc 23 reçoit
  directement la perte velocity globale. Le polish ajuste des paramètres FP16.
- **Les scales ne sont pas des paramètres appris indépendants.** Elles sont
  recalculées à partir des poids ; le STE transmet un gradient identité.
  L'erreur relative des poids maîtres quantifiés, proche de 0,48 au bloc 23,
  n'est ni une erreur audio ni une borne sur la capacité optimale.
- **Configuration et exécution divergent.** Le JSON v4 annonce LR 2e-5,
  weight decay 0 et accumulation 4. Le trainer utilise 5e-5 vers 5e-6,
  weight decay 1e-4, sans accumulation implémentée. Les sigmas viennent du
  sampler à huit pas, pas de la grille du JSON.
- **L'indépendance est insuffisamment prouvée.** Le parent est un hash du nom
  de source, pas une identité d'œuvre/session ni une déduplication PCM.
  L'ancien test a été consulté. Il devient un jeu de régression de développement.
- **Les trajectoires peuvent rassurer à tort.** L'audit conserve les états
  avant mise à jour, pas le latent terminal. Un cosinus élevé d'états encore
  dominés par le bruit commun ne prouve pas une bonne génération.

La comparaison v3/v4 ne mesure pas proprement un effet « davantage de données » :
validations différentes, peu de nouveaux parents nommés, doublons possibles,
un seul seed. Les fenêtres terminales déjà essayées ne valent pas plusieurs
passes globales sur tous les blocs avec un nouveau quantizer.

Sources locales : [rapport v4](TERNARY_V4_AUTHORIZED_EXECUTION_REPORT_2026-09-22.md),
[registre historique](TERNARY_RECOVERY_EVIDENCE_2026-09-22.md),
[base de connaissances](TERNARY_DISTILLATION_KNOWLEDGE_BASE.md).

## 3. Définition du résultat et levier de taille

### Périmètre final

Cible : **toutes les matrices apprises du DiT en codes ternaires symétriques**,
activations FP16, calculs sensibles et états d'optimiseur FP32.

Pour chaque groupe : `q ∈ {-1,0,+1}`, `W_quant = scale * q`.
Scales partagées positives ; groupe nul traité explicitement.
Pas de moyenne affine ajoutée aux poids, de quatrième code, de somme de
plusieurs matrices ternaires ou de résidu LoRA dense dans le résultat final.

Inclure attention, FFN, conditionnement local/global/temporel, entrées/sorties,
convolutions et tokens mémoire appris. Inventorier séparément le petit
conditionneur de durée livré dans le checkpoint. Biais vectoriels,
normalisations, gates, scales et buffers gardent leur précision déclarée.
Ce contrat ne signifie pas « tous les scalaires du pipeline sont -1/0/+1 ».
T5 et SAME-L restent hors du DiT et doivent être comptés séparément.

Une variante Hadamard porte explicitement le nom « ternaire en base tournée » :
ses codes sont ternaires, ses poids équivalents dans la base d'origine sont
généralement denses. La voie sans rotation reste le contrôle strict.

### Inventaire réellement lu le 23 septembre

Checkpoint :
`/Users/guillaumegaillard/.cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz`.

Lecture des en-têtes NPY, sans charger de modèle :

- 525 tenseurs ; 1 453 368 336 éléments ; 2 907 132 960 octets de tenseurs.
- Core historique : 168 matrices, 1 358 954 496 poids.
- Tenseurs de dimension ≥2 : 230, soit 1 452 609 536 éléments.
  Ce décompte inclut `cond.seconds_total_weight`, à classer séparément du DiT.
- Les poids core représentent environ 93,5 % des éléments ; couvrir le reste
  est à la fois une difficulté qualité et un levier de stockage.

Estimation arithmétique, **pas fichiers exportés** : toutes ces matrices,
padding de chaque ligne au multiple du groupe, slots 2-bit et un scale FP16
par groupe ; autres tenseurs conservés à leur dtype du checkpoint.

| Groupe | Codes | Scales | Autres tenseurs | Total calculé |
|---|---:|---:|---:|---:|
| 32 | 363 438 080 | 90 859 520 | 1 520 672 | 455 818 272 octets |
| 64 | 363 732 992 | 45 466 624 | 1 520 672 | 410 720 288 octets |
| 128 | 364 322 816 | 22 770 176 | 1 520 672 | 388 613 664 octets |

Ajouter conteneur, manifeste, buffers absents du checkpoint et éventuels
signes/padding de rotation. Le chargement MLX peut reconstruire un biais par
groupe et consommer davantage. Aucun gain ZIP supposé dans ce calcul.

**Conséquence : G32 intégral paraît compatible avec 500 Mo.** Le jalon
core-G32 peut dépasser 500 Mo et rester utile à la recherche ; il ne devient
pas un produit livré. Ne plus rejeter ce groupe sur le seul export partiel.

Conserver 500 000 000 octets comme cible finale, pas 500 MiB. L'ancien chiffre
455,8 Mo n'est toujours pas un modèle de qualité livré.

## 4. Recherche vérifiée et décisions

Revue ciblée au 23 septembre 2026 ; transfert vers SA3 explicitement expérimental.

- **[CAT-Q, 25 juin 2026](https://arxiv.org/abs/2606.26650)** :
  conversion de LLM préentraînés, modulation apprise, transition douce/dure,
  optimisation par fenêtres. Son décalage intervient dans l'assignation ;
  il n'est pas ajouté au poids déployé. Recette publique à étudier dans
  [BitTern](https://github.com/IntelChina-AI/BitTern).
  Retenir une adaptation contrôlée, pas la promesse qu'un corpus audio minuscule
  suffira. Les résultats conservent un écart au FP16.
- **[RobuQ, révision du 21 mai 2026](https://arxiv.org/html/2509.23582v2)** :
  QAT ternaire de DiT image préentraînés, rotations et activations adaptées.
  Important : 350 000 itérations, batch 8, RTX A6000 48 Go dans le protocole
  principal ; embeddings et couche finale gardés en pleine précision.
  Cela appuie la faisabilité de conversion, pas « tout SA3 sous 12 Go en 200 pas ».
  Ne pas ajouter la quantification des activations à notre première difficulté.
- **[PTQ pour Audio DiT, septembre 2025](https://arxiv.org/abs/2510.00313)** :
  Stable Audio Open W8A8/W4A8, sensibilité aux canaux et timesteps.
  Retenir couverture temporelle et contrôles audio ; leur branche résiduelle
  continue n'est pas notre résultat ternaire strict.
- **[ParetoQ, février 2025](https://arxiv.org/abs/2502.02631)** :
  aux très faibles précisions, apprentissage des scales et réorganisation des
  représentations comptent. Justifie de comparer quantizers et budgets réels,
  pas d'imposer une proximité élément par élément avec les poids initiaux.
- **[TerDiT](https://arxiv.org/html/2405.14854v2)** :
  démontre des DiT image ternaires entraînés depuis zéro et une sensibilité
  du conditionnement. Référence de faisabilité ; pas raison de jeter notre
  préentraînement ni d'ajouter une normalisation qui changerait la baseline.
- **[Bonsai Image](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-unpacked)**
  et **[Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)** :
  artefacts image/LLM low-bit et exécution spécialisée. Distinguer poids
  dépackés, packing et kernel. Bonsai 2 utilise H1024 et groupes 128 :
  taille de rotation et taille de groupe ne sont pas synonymes.
- **[Ternary Mamba, juin 2026](https://arxiv.org/abs/2606.18114)** :
  exemple d'effondrement de l'occupation des codes avec scales apprises.
  Architecture différente ; alerte utile pour surveiller les zéros, pas
  interdiction générale d'apprendre les scales.
- **[TASQ, août 2026](https://arxiv.org/abs/2608.03057)** :
  bits variables par timestep avec buffer de poids à précision maximale.
  Intéressant pour le calcul ; ne résout pas notre contrat de stockage ternaire.

Aucun travail consulté ne reproduit notre combinaison exacte SA3 Medium,
scope intégral, 500 Mo, mémoire de 12 Go et qualité audio.

## 5. Méthode retenue

### 5.1 Quantizer exportable, appris avant de multiplier les runs

Conserver un oracle historique G64 absmean. Construire à côté un quantizer
symétrique avec scale et seuil appris ; challenger : modulation de
l'assignation inspirée de CAT-Q. Le décalage éventuel choisit les codes,
il ne réapparaît jamais comme moyenne additive à l'inférence.

Le quantizer définit explicitement son gradient, ses bornes, son calendrier
de durcissement et le traitement des groupes nuls. Portage de la fonction
douce depuis une révision figée du code de référence ; comparaison sur
petits tenseurs avec un oracle indépendant. Pas d'approximation inventée
puis annoncée comme reproduction de CAT-Q.

Séparer deux modes :

1. Calibration légère : poids préentraînés fixes, paramètres de quantification
   appris dans la fenêtre active.
2. Récupération QAT : réouverture des poids maîtres FP32 de cette fenêtre,
   avec leur état d'optimiseur, si la calibration plafonne.

La relaxation douce est un outil d'entraînement seulement. Chaque validation
exécute les codes durs et scales arrondies à la précision de déploiement.
Réserver au moins le dernier quart du budget à la quantification dure ;
prolonger cette phase si l'écart doux/dur reste élevé. Sauver séparément
meilleur checkpoint dur et checkpoint reprenable.

Pas de trois logits FP32 par poids : coût inutilement élevé pour ce budget.
Surveiller fractions -1/0/+1, groupes morts, changements de codes et dérive
des scales ; aucun quota universel de zéros imposé sans ablation.

Point de départ du pilote, non réglage validé : microbatch 1, accumulation 4,
AdamW à états FP32, weight decay 0, epsilon 1e-8, clip global 1.0.
LR des maîtres 1e-5 ; paramètres de quantification sans dimension 3e-4,
avec gradient scaling testé. Warm-up 5 %, décroissance finale d'un facteur 10.
Garder ces groupes d'optimisation distincts ; ne pas appliquer epsilon 1e-8
à des moments FP16. Comparer un seul changement de LR si instabilité.

### 5.2 Contrat de rotation, si elle gagne le pilote

Pour activations en lignes et matrice orthogonale R :

```text
W_rot = W @ R
Q = ternarize(W_rot)
y = (x @ R) @ Q.T
```

Ici Q inclut les scales. Tester d'abord l'identité sans quantification.
La même base, les mêmes signes, la normalisation et le padding doivent être
utilisés pendant apprentissage, export et inférence.

Première ablation H128, pas recherche illimitée de rotations.
L'entrée locale 257 exige padding de x et W, puis conservation des bonnes
dimensions. Tester le chemin avec conditionnement non nul.
Aucune traversée implicite de SiLU, RMSNorm, RoPE ou connexions résiduelles.

### 5.3 Objectif final, pas seulement sorties des blocs

Sur le même état x, timestep t et condition c :

```text
Lv = mean((student(x,t,c) - teacher(x,t,c))²)
     / max(mean(teacher(x,t,c)²), epsilon)

L = Lv + 0.1 * (1 - cosine_velocity) + lambda_local * Llocal
```

Calcul des réductions FP32 ; epsilon fixé à partir du plancher mesuré.
`lambda_local=0.1` au départ, puis 0 en finition : valeurs de départ,
à comparer sur développement. La MSE conserve l'information d'amplitude,
contrairement au cosinus seul. Ne pas normaliser arbitrairement les sorties.

En calibration locale, comparer aussi le résidu du bloc, pas seulement
`h + residual`, dont la connexion identité peut masquer l'erreur.
Le teacher local reçoit la même entrée de fenêtre que le student.
La perte globale compare les DiT complets, tous deux appelés sur le même x.

Pour le checkpoint d'inférence distillé, ne pas remplacer cette cible teacher
par une loss rectified-flow BASE supposée équivalente. Vérifier ARC/BASE,
paramétrisation, sampler et conventions avant toute loss auxiliaire.

D'abord distillation sur états cachés en cache, puis rafraîchissement depuis
des trajectoires étudiantes du train. Interroger le teacher sur ces états,
pas sur sa propre trajectoire différente pour calculer Lv. Un rollout
différentiable à deux pas est une ablation ultérieure, pas un prérequis.
Une perte à un seul pas équivalente à Lv repondérée n'est pas un signal nouveau.

## 6. Exécution, dans cet ordre

### P0 — rendre l'expérience fiable

- Figer hashes des poids, buffers, runtime, codec et texte ; aucune LoRA pour
  la première référence. Reproduire le teacher dans le chemin d'audit.
- Une configuration réellement consommée par le trainer ; champs inconnus
  refusés. Tester LR, accumulation, weight decay, crop, durée et sigmas effectifs.
- Corriger audit terminal et distinction erreur de fonction/erreur de trajectoire.
  Auditer export dans un processus neuf, pas seulement l'objet en mémoire.
- Tests : packing/unpacking exact, code réservé refusé, zéro exact, scales
  FP16, padding 257, Conv1d, tokens mémoire, fréquences temporelles FP32,
  couche manquante refusée, gradients à travers suffixe gelé.
- Vérifier reprise interrompue contre exécution continue à RNG et minibatches
  identiques. Ne jamais reconstruire un état QAT depuis les seuls codes exportés.
- Contrôles FP16 et W4A16 sur les mêmes cas. W4A16 est un diagnostic non
  promouvable ; s'il échoue également, isoler la chaîne avant le ternaire.

Livrables : `baseline_manifest.json`, `scope.json`, `resolved_config.json`,
`numerical_contract_report.json`. Noms futurs, pas fichiers déjà produits.

### P1 — données utiles et vrai test neuf

Autorisation utilisateur acquise pour étude personnelle locale des corpus.
La consigner une fois avec périmètre ; ne pas en déduire droits tiers vérifiés,
redistribution ou autorisation permanente pour tout futur dataset.

1. Grouper par œuvre/enregistrement/session ; hashes fichier et PCM, dérivés
   et quasi-doublons examinés ensemble. Des noms différents ne suffisent pas.
2. Reclasser les anciens validation/test en développement historique.
   Constituer un nouveau test, exclu du choix de méthode et de checkpoint.
3. Distinguer captions vérifiées, descriptions incertaines et prompts de
   synthèse. Pour distillation, le teacher définit la cible conditionnelle ;
   une étiquette de fichier ne devient pas une vérité musicale.
4. Petit cache initial : 512 états, au moins 64 parents train si disponibles,
   diversité de familles et de niveaux de bruit. Étendre à 2 048 états, puis
   8 192 uniquement si les courbes justifient le coût. Un état n'est pas une
   nouvelle source audio.
5. Mélange initial : moitié latents réels bruités, moitié trajectoires teacher.
   Après stabilité, 25 % réels, 25 % teacher, 50 % états student rafraîchis.
   Proportions expérimentales ; conserver un contrôle sans états student.
6. Couvrir les huit timesteps exacts de production ET une grille stratifiée
   sur [0.01, 0.99]. Journaliser les populations séparément, y compris bas sigma.
   Inclure graines nouvelles, plusieurs durées et conditions vides si utilisées.

Validation cible : ≥32 parents et ≥24 prompts, avec cas connus et nouveaux
assemblages de descriptions. Test final : ≥48 parents, ≥24 prompts, deux seeds
minimum ; échantillons distincts par parent. Si insuffisant : résultat pilote
seulement, collecte ciblée ou prompts/trajectoires synthétiques séparés, sans
revendiquer généralisation sur sources réelles.

Caches : stocker embeddings texte bruts et conditions d'entrée. Dès qu'une
projection de conditionnement est quantifiée, recalculer ses sorties côté
student. Réutiliser les projections du teacher court-circuiterait le modèle testé.

Livrables : manifestes avec `content_hash`, `pcm_hash`, `lineage_group`,
`split`, `annotation_status`, `usage_authorization`, exposition historique
et rapport de fuite. Provenance du préentraînement teacher inconnue déclarée.

### P2 — choisir une méthode par un pilote apparié

Trois fenêtres représentatives : blocs 0–1, 11–12 et 22–23, reste FP16.
Même cache, budget, seed, forward final et sorties audio pour chaque variante.

Ordre borné :

1. Quantizer G64 historique, reconstruit depuis le même teacher dans chaque
   fenêtre et entraîné au même budget court ; pas une ancienne métrique réutilisée.
2. Quantizer appris G64, sans rotation.
3. Meilleure méthode G32.
4. H128 sur cette méthode, seulement si encore nécessaire.

10 puis 50 mises à jour servent au profil mémoire et aux gradients, pas au
verdict qualité. Pilote de 250 mises à jour par fenêtre ; extension à 1 000 si
validation dure progresse. Accumulation effective 4 à vérifier, donc compter
séparément minibatches, mises à jour et expositions par source.
Confirmer les deux finalistes avec un second seed ; arrêter l'exploration
des variantes après cette comparaison.

Avant chaque extension, publier coût mesuré et enveloppe de calcul restante.
Un budget épuisé signifie expérience incomplète, pas preuve d'impossibilité.

Sélection : réduction appariée de Lv globale en quantification dure, dispersion
par parent/sigma, conservation d'amplitude et absence de régression audio
récurrente. Viser ≥20 % d'erreur en moins que l'assignation historique sur
ces fenêtres ; cible d'ingénierie, pas preuve de qualité du DiT complet.
En cas de résultat ambigu, conserver l'incertitude ; aucun minimum global
arbitraire à atteindre après seulement 50 pas.

### P3 — convertir le core et le récupérer globalement

Initialisation depuis le teacher intact, pas depuis l'export v4 dégradé.
Réutiliser v4 uniquement comme contrôle de développement.

1. Convertir par fenêtres de deux blocs ; garder le reste exécutable.
   Optimiser Lv à travers le suffixe dès cette phase, avec checkpointing
   d'activations. Le préfixe gelé peut être détaché ; le suffixe ne doit pas
   couper le gradient vers la fenêtre.
2. Faire un premier passage complet, puis au moins un passage de réouverture
   de tous les blocs. Alterner paires 0–1/2–3/... puis fenêtres décalées avec
   traitement des bords. Chaque changement amont invalide ses caches aval.
3. Budget de travail initial : 500 mises à jour par paire pour la conversion,
   puis 250 par fenêtre de récupération ; conserver les courbes et adapter
   seulement sur validation. Ces milliers d'updates ne garantissent rien.
4. Après chaque passage, recalculer états student, export dur, audit global et
   trois rendus canaris fixes. Ne pas ajouter uniquement des steps au bloc 23.
5. Finition éventuelle : scales de tous les blocs optimisées ensemble, codes
   fixés ; puis réouverture des codes si le gain plafonne. Une calibration
   globale des scales n'est pas un entraînement joint de tous les poids.

Candidat core : jalon intermédiaire, même s'il dépasse 500 Mo. Pas de
publication ni de substitution au modèle de référence.

### P4 — ternariser les matrices restantes sans perdre le gain

Traiter les familles une à une, chacune suivie d'une récupération globale :

- matrices de conditionnement local, incluant entrées 257 ;
- projections de texte, durée, timestep et modulation globale ;
- projections d'entrée/sortie et convolutions ;
- tokens mémoire appris et éventuels tenseurs matriciels oubliés.

L'ordre définitif vient de la sensibilité mesurée, pas d'une liste intuitive.
Les matrices sensibles reçoivent G32 avant d'envisager une précision supérieure.
Un contrôle FP16 temporaire est permis pour isoler un défaut ; il ne termine
pas la branche « toutes matrices ternaires ».

Attention aux chemins dormants : zéro conditionnement local n'exerce pas
tous les poids. Si audio-to-audio ou inpainting font partie du résultat,
inclure des conditions/masks non nuls dans calibration et validation.

Après extension : audit du forward réellement exécuté, non simple liste de
clés du NPZ. Toutes les matrices du scope doivent venir du paquet ternaire.
Aucun fallback teacher ni projection continue cachée.

### P5 — paquet final, qualité et décision

- Export depuis un snapshot dur unique : aucun second quantizer au packing.
- Compactage mono-scale avec biais MLX dérivé ; round-trip exact des codes
  et scales. Packing cinq trits/octet seulement si nécessaire, après qualité,
  avec kernel/loader mesurés. Il ne crée aucune capacité supplémentaire.
- Recharger sans poids maîtres, optimiseur ni teacher accessibles au loader.
- Évaluer uniquement ce paquet. Enregistrer tous les résultats, pas seulement
  meilleur seed ou meilleur prompt.
- Geler configuration et hash candidat avant ouverture du nouveau test.
  Si le test sert ensuite à corriger, le reclasser en développement et réserver
  un autre test final.

## 7. Budget matériel et sauvegardes

Limites maintenues : 12 000 000 000 octets accélérateur, garde à
11 000 000 000. Mesurer Metal, mémoire physique du processus, pression système
et delta de swap ; pas addition aveugle RSS + Metal. Le swap déjà présent
avant le run ne prouve pas à lui seul un dépassement causé par ce run.
Un résultat Mac ne certifie pas une carte CUDA de 12 Go.

QAT globale Adam FP32 naïve : poids + gradients + deux moments donnent
16 octets par paramètre, soit environ 23,25 Go pour 1,45 milliard de paramètres,
avant activations et teacher. Elle ne tient pas telle quelle.

Voie locale :

- une seule fenêtre de 1–2 blocs avec maîtres/gradients/moments FP32 ;
- teacher exécuté séparément pour le cache, puis déchargé ;
- T5 et codec hors mémoire pendant QAT ;
- poids hors fenêtre packés, activations recomputées ; gradient d'entrée du
  kernel quantifié vérifié contre un oracle dense ;
- si kernel non différentiable : déquantification temporaire d'un bloc à la
  fois avec recomputation, jamais tous les poids maîtres simultanément ;
- aucun « offload CPU » supposé libérer de la RAM sur mémoire unifiée.

Le core d'un bloc représente environ 0,906 Go d'états Adam FP32, deux blocs
1,812 Go, hors paramètres auxiliaires et activations. C'est une estimation,
à confirmer par 10/50 updates puis un cycle complet teacher/cache/student.

Checkpoint reprenable : maîtres, quantizer, optimiseur, RNG, accumulation,
position dans les données, hashes, fenêtre et calendrier doux/dur.
Sauvegarde sharded ; ne pas recopier tout l'historique à chaque 50e update.

État disque observé : environ 7,7 GiB libres en interne, 37 GiB sur
`/Volumes/Extreme SSD`. Réserve interne de 15 GiB non satisfaite.
Avant run long : prévoir espace supplémentaire ou déplacement explicitement
choisi ; aucune suppression automatique. Premier pilote : ≤8 GiB de nouveaux
dérivés, sur SSD externe, puis réestimation. Conserver les artefacts rejetés.

Temps et coût : mesurer secondes/update pour chaque mode. Publier
`cache + nombre_updates × secondes_update + validation + exports`, avec marge,
avant entraînement long. Les 89,5 minutes du v4 local ne prédisent pas le coût
d'une perte globale. Aucun GPU/API payant lancé implicitement.

## 8. Critères de réussite

Critères à figer sur développement avant le test :

1. **Structure** : couverture complète du scope annoncé ; trois codes seulement ;
   aucune moyenne de groupe ajoutée ni branche dense cachée.
2. **Numérique** : codes/scales identiques au reload ; forward de référence et
   forward packé dans l'enveloppe d'erreur de réduction mesurée. Cible relative
   1e-3 maximum pour kernels différents ; parité exacte attendue pour le même
   chemin. Vérifier chaque famille et le modèle, plusieurs durées/sigmas.
3. **Fidélité** : Lv, cosinus, ratios RMS et cas défavorables par parent/sigma.
   Les repères historiques 0,93 moyen/0,85 minimum restent provisoires ;
   ils ne suffisent ni à certifier ni à réfuter seuls la qualité musicale.
   Comparer aussi trajectoires complètes et latent terminal.
4. **Audio** : d'abord 3 canaris, puis ≥24 paires à seeds appariés, 12/30 s,
   et ≥3 rendus longs à durée d'usage. Mesurer deux canaux, dynamique,
   transitoires, grave, aigus, stéréo, silence et répétitions.
   WAV float brut avant gain ; un pic float >1 n'est pas déjà du clipping PCM.
5. **Écoute** : A/B masqué à niveau égal par l'utilisateur, mêmes paramètres.
   Noter adhérence au prompt, artefacts, timbre, dynamique et structure ;
   absence de dégradation récurrente gênante requise. Documenter votes,
   désaccords et cas rejetés ; pas de faux verdict humain automatique.
6. **Taille et ressources** : paquet final ≤500 000 000 octets ; limites mémoire
   respectées aux phases réellement testées, chargement et décodage inclus.
   Publier également dépendances, taille décompressée et temps d'inférence.

Aucun EQ, limiteur ou débruitage pour masquer les défauts. Copies d'écoute :
gain constant documenté et marge true-peak seulement. Un bon cosinus, une
bonne MSE ou un gate PCM vert ne remplace pas l'écoute.

Statuts distincts : `software_validated`, `core_candidate`,
`all_matrices_candidate`, `technical_pass`, `musical_review_pending`,
`accepted_personal_use`. Seul le dernier, avec périmètre et budgets conformes,
répond à l'objectif pratique.

## 9. Si une phase échoue : prochain essai utile

- FP16/W4 échouent : corriger baseline/loader/sampler, pas augmenter le corpus.
- Local améliore, global stagne : revisiter tous les blocs et gradients du
  suffixe ; augmenter exposition sur états student ou faible sigma.
- Doux bon, dur mauvais : allonger la phase dure et revoir gradient/seuil ;
  ne jamais exporter la relaxation comme modèle ternaire.
- Zéros/scales divergent : borner paramètres, réduire leur LR, puis contrôle
  scales statistiques. Ne pas ajouter une pénalité arbitraire sans mesure.
- Certaines matrices dominent : groupes plus fins et récupération ciblée ;
  comparer rotation appariée. Pas de résidu dense accepté en secret.
- Train s'améliore, validation régresse : nouvelles lignées/durées/styles,
  cache renouvelé et moins de répétitions ; pas simplement plus de fichiers.
- Plateau confirmé après deux expériences ciblées : chiffrer davantage de QAT
  ou une autre optimisation. Nouveau budget nécessaire avant GPU payant.
  Entraînement natif depuis zéro reste une autre recherche, pas un raccourci.

Un échec réduit une hypothèse. Il ne justifie ni un redémarrage identique,
ni l'abandon global du ternaire, ni une promesse de réussite au run suivant.

## 10. Travail d'implémentation à réaliser

Réutiliser contrat, checkpointing et outils existants ; corriger leurs limites
au lieu de créer un nouveau script monolithique par variante.

- `prepare_ternary_dataset.py` et `build_ternary_independent_corpus.py` :
  lignées, exposition historique, vrais splits et budgets de caches.
- `ternary_contract.py` : snapshot unique, groupes par matrice, padding,
  rotation optionnelle, dispatch Linear/Conv/mémoire et loader strict.
- `train_ternary_quality.py` : configuration effective, quantizer appris,
  phase douce/dure, accumulation et logs d'occupation.
- `finetune_ternary_window.py` : passages réouvrant tous les blocs,
  maîtres persistants, gradient du suffixe et rafraîchissement des caches.
- `audit_ternary_quality.py` : manifeste de split vérifié, seeds/parents
  complets, teacher sur état student, latent terminal et comparaison appariée.
- `render_ternary_quality.py` et `measure_ternary_resources.py` :
  audit stéréo brut, durées d'usage, pression mémoire et swap différentiel.
- Tests : paramètres de configuration appliqués, gradients, reprise,
  équivalence doux désactivé/hard/export, couverture dynamique et anti-fuite.

Les pilotes P0/P1/P2 ont depuis été exécutés. Voir le journal d'exécution §11.
La cascade complète reste conditionnée à des contrôles de qualité par fenêtre ;
les pilotes seuls ne justifient pas une acceptation du modèle.

## 11. Journal d'exécution v6 — 23 septembre 2026

Ce journal sépare les faits mesurés des hypothèses. Les audits mentionnés ici
réutilisent un split de développement déjà consulté lors de travaux antérieurs ;
ils ne constituent pas un test final scellé ni une preuve d'indépendance des
œuvres/sessions.

### Cause racine du précédent entraînement sans apprentissage

La recomputation MLX utilisait `mx.checkpoint(apply_layer)` avec le module
capturé dans une closure. Dans cette forme, les paramètres du module n'étaient
pas des entrées explicites de la fonction différentiée : le test minimal et le
run d'audit ont observé des gradients nuls sur poids et seuils. Le run ne doit
donc pas être considéré comme QAT valide, même si sa loss s'affichait.

Correction dans `services/musicgen/train_ternary_window_v6.py` : transmettre
`module.trainable_parameters()` en entrée de la fonction checkpointée, puis
mettre à jour explicitement le module depuis cet argument avant le forward.
Une régression vérifie désormais des gradients non nuls des poids et du seuil.
Le run corrigé a mesuré une norme de gradient de seuil de 0,01809 et un delta
moyen de paramètres de 0,000797 après une mise à jour. Cette forme suit le
pattern public utilisé par [mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py).

Un second écart affectait le hard-freeze : l'affectation des codes ternaires
utilisait un seuil appris, tandis que l'export réutilisait la scale de
reconstruction comme seuil. Le contrat partagé implémente maintenant
l'affectation symétrique avec le multiplicateur de seuil appris borné, et le
test compare le forward quantifié avant/après export. Le pilote appris n'a pas
surpassé le contrôle symétrique fixe ; G32 symétrique fixe est donc le choix
provisoire, non une conclusion universelle.

### Cache enseignant, mémoire et tests

La configuration en ligne gardant l'enseignant en mémoire a atteint environ
11,65 GiB et a été interrompue sans écrire de checkpoint partiel. Le cache
fp16 des cibles du professeur contient 512 états, calculés en environ 197 s,
avec un pic d'environ 3,67 GiB. Fingerprints et états sont vérifiés avant
utilisation. Sur une première mise à jour, perte en ligne et perte en cache
différaient d'environ 1,5e-6. En QAT avec cache, pics observés : 6,02 GiB pour
G32 symétrique fixe et 8,94 GiB pour le seuil appris. Ces chiffres décrivent
les runs testés, pas une garantie pour l'audit audio ou le modèle entier.

Seize tests ciblés (contrat, parité hard-freeze/export, reload, provenance du
cache et gradients/checkpointing) ainsi que la compilation Python ont passé
avant la cascade suivante. À relancer après toute modification pertinente.

### Pilotes correctifs G32, fenêtres 0–1, 11–12 et 22–23

Chaque fenêtre a reçu 250 mises à jour et a rechargé son artefact sans
différence numérique. La taille des artefacts de fenêtre n'est pas la taille
du modèle complet. Les nombres ci-dessous proviennent de l'audit des 18
prompts, 5 sigmas et rollouts de 8 pas sur le split de développement :

| Fenêtre | Cosinus velocity moyen / min. | Erreur relative moyenne | Cosinus état final moyen | Gates audit |
| --- | ---: | ---: | ---: | --- |
| 0–1 | 0,96585 / 0,71609 | 0,24626 | 0,90053 | échec |
| 11–12 fixe | 0,97419 / 0,86541 | 0,20642 | 0,87300 | passe |
| 22–23 | 0,95823 / 0,90507 | 0,28262 | 0,95768 | passe |

La tête échoue sur son pire point (sigma 0,95, cosine 0,71609, erreur
relative 0,77373), malgré un ratio RMS de 1,0499 : ce n'est pas l'effondrement
d'amplitude observé lors d'essais antérieurs. Il faut améliorer le pire cas de
la tête avant toute acceptation. Au milieu, le seuil appris a obtenu
0,97362/0,85381, un peu moins bien que le seuil fixe ; environ 0,68 % des codes
changeaient et aucune saturation des bornes de seuil n'a été détectée.

### Prochaine opération et statut

Le test séquentiel 2–3 a été exécuté depuis les records 0–1, G32 symétrique,
cache enseignant, 250 mises à jour. Son cumul 0–3 passe le contrat/reload,
mais **échoue au gate de qualité** : moyenne/minimum velocity cosine
0,95104/0,72228, contre 0,96585/0,71609 pour 0–1 seul ; cosine terminal
moyen 0,88324 contre 0,90053. Les moyennes par sigma baissent de 0,015 à
0,027 ; le minimum ne monte que légèrement sur le même prompt difficile,
sigma 0,95. Ratios RMS proches de 1 indiquent une erreur surtout directionnelle,
pas un collapse d'amplitude. Le train loss baisse à 0,07054, sans transfert
de validation suffisant. **Ne pas avancer à 4–5 en gardant 0–1 gelé.**

Cause exploitable : la cascade stricte ne permettait pas de réouvrir les
records source d'une fenêtre déjà ternarisée. Le trainer v6 sait maintenant
les convertir en poids maîtres déquantifiés, avec parité forward vérifiée,
et réentraîner toute fenêtre sélectionnée. Smoke conjoint des blocs 0–3 :
28 modules rouverts, update réussie, loss 0,10620, pic 7,45 GiB.

La passe conjointe 0–3 (250 updates, 28 modules rouverts) a légèrement amélioré
le cumul séquentiel, mais sans récupérer le gate : velocity cosine
0,95216/0,72978, erreur relative moyenne 0,29407, cosine terminal moyen
0,88582. Par rapport au cumul séquentiel, gain de seulement +0,00112 en
cosine moyen et +0,00750 au pire point ; le candidat exige ≥0,90/0,80.
Worst prompt et sigma inchangés (rekorder-3.2, 0,95), RMS ratio 1,04189.
Artefact cumulatif de 2 324 603 392 octets, 28 matrices, reload exact.
Le run s'est terminé avec loss 0,06766 et pic Metal 8,71 GiB. Dix-huit tests
ciblés et compilation repassent après le support de réouverture.

### Validation indépendante et récupération on-policy — 23 septembre 2026

Le corpus v6 est désormais un split source/prompt/hash disjoint : train
434 latents / 383 parents / 85 prompts (dont 181 nouveaux exemples SFT
répartis sur 57 prompts), validation 33 / 33 / 12, test 41 / 41 / 24.
L'audit a sélectionné les 12 prompts de validation. Le test de 24 prompts
n'a été ni entraîné ni audité et reste scellé. Le manifeste de sélection
imbrique les chemins sous `validation.sources[].staged_latent` ; l'auditeur a
été adapté à ce schéma et un test de régression a été ajouté. Les 21 tests
ternaires ciblés et la compilation Python des outils modifiés passent.

Trois raffinements ont été exportés puis rechargés exactement (G32 symétrique,
28 projections, blocs 0–3 rouverts, 250 updates chacun). Tous les audits
ci-dessous utilisent les mêmes 12 prompts, seed 4242, cinq sigmas, rollout
stochastique 8 pas. Les deux premières colonnes évaluent les sorties
ponctuelles teacher/student ; les colonnes terminales comparent les trajectoires.

| Checkpoint sur validation v6 | Cosinus ponctuel moyen / min. | Gate ponctuel candidat / release | Cosinus d'état terminal moyen / min. |
| --- | ---: | --- | ---: |
| Cache élargie 2048, ancien corpus (contrôle apparié) | 0,93584 / 0,80593 | passe / échec | 0,61383 / 0,45651 |
| SFT indépendant + états teacher | 0,94239 / 0,84282 | passe / échec | 0,64923 / 0,51250 |
| Refresh sur trajectoires student, 8 pas | 0,94431 / 0,79815 | échec / échec | 0,66628 / 0,57049 |
| Refresh student 16 pas, sigma pondérés | 0,94813 / 0,87302 | passe / passe | 0,66579 / 0,54982 |

Le refresh student 8 pas améliore le terminal moyen de 0,017 et le pire
terminal de 0,058, mais manque de peu le gate ponctuel candidat (minimum
0,79815 pour un seuil de 0,80). Le cache suivant a sur-échantillonné sigma
0,95 et les états proches de 0,50/0,274, avec 1 360 états bruités et
1 360 états student sur 16 pas, couvrant les 85 prompts train. Il récupère
le gate ponctuel de release (≥0,93/0,85), mais ne récupère pas les trajectoires :
terminal 0,66579/0,54982, pratiquement inchangé face au refresh 8 pas.

Le désaccord commence tard dans le sampler, pas sur l'état initial : cosinus
moyen des états 0,9982 au pas 4 (σ=0,891), 0,9792 au pas 5 (σ=0,746),
0,8779 au pas 6 (σ=0,512), puis 0,7306 au pas 7 (σ=0,274). Le terminal
moyen tombe à 0,6658. Le pire prompt terminal de cette passe est
« voice, a-cappella; 136 BPM » (cosinus 0,5498, ratio RMS 0,8802).
Le biais vers les seuls états teacher était donc une partie du problème,
mais le simple teacher-forcing sur des états student et davantage d'états
n'améliorent pas assez l'erreur accumulée du sampler.

La dernière passe a une loss train 0,07841 → 0,10745 (bruitée, sans baisse
finale), pic Metal QAT 8,74 Gio ; le cache teacher fp16 `[2720,256,128]`
a culminé à 3,69 Gio. Son export partiel fait 2 324 603 402 octets et son
reload est exact. Ces 28 matrices sont seulement le core de quatre blocs :
les 20 blocs restants sont denses. Ce n'est donc ni le modèle entièrement
ternaire ni une validation de la cible 455,8 Mo.

**Décision : ne pas étendre à 4–5.** Le gate ponctuel est enfin vert, mais les
trajectoires restent très éloignées du teacher et aucun audio décodé/écouté
n'a été validé. Prochain essai utile : loss de rollout différentiable sur
deux pas tardifs du sampler (départ autour de σ=0,512, puis σ=0,274), comparée
au teacher sur le même bruit et les mêmes prompts train. Garder l'entraînement
sur train, utiliser validation uniquement pour choisir, et ne déverrouiller
le test qu'une fois le checkpoint et le protocole figés.

Le core reste **incomplet**. Les phases suivantes sont suspendues jusqu'à ce
qu'une loss de rollout tardif améliore réellement le terminal : seulement
ensuite reprendre les paires séquentielles, l'audit cumulatif, le contrôle G64,
la seconde seed, la couverture structurelle complète, les reloads multi-durée,
les rendus audio et l'écoute A/B. Les seuils de ce plan restent des gates de
développement ; la validation v6 a déjà servi à choisir les variantes et ne
peut plus servir de test final. Garder les 24 prompts de test scellés jusqu'au
gel du modèle et du protocole d'évaluation.
