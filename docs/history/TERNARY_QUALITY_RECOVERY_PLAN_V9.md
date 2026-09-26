# Plan V9 révisé — ternariser Stable Audio 3 au périmètre Bonsai

Révision : **9.1, 25 septembre 2026**. Plan actif.
Statut : **P0/P1 clôturés ; pilote P2 G32 techniquement réussi sur [0,1]
avec deux seeds et sur le bloc 2 avec rollout on-policy ; revue audio
humaine en attente ; P3 non lancé**.

Décision utilisateur : **« ternariser comme Bonsai, pas plus »**.
L'objectif toutes-matrices de la V9 initiale est abandonné, pas reporté à une
phase suivante. L'[ancienne V9](archive/TERNARY_QUALITY_RECOVERY_PLAN_v9_initial_2026-09-25.md)
reste disponible pour comprendre l'historique.

## 0. Résultat recherché et décisions fermes

Produire un DiT Stable Audio 3 Medium dont les grandes matrices d'attention
et de FFN utilisent des codes `{-1, 0, +1}` avec une scale par groupe,
en conservant les supports sensibles à leur précision native. Le modèle
doit générer de l'audio utile, comparable au modèle de référence, sans
correction acoustique destinée à masquer un défaut.

Trois priorités, dans cet ordre :

1. Préserver la qualité musicale et les capacités effectivement évaluées.
2. Convertir **tout le cœur attention/FFN**, pas seulement quelques couches.
3. Réduire le stockage et mesurer les ressources réelles.

Le maintien des supports flottants est **la cible finale autorisée**,
pas un témoin provisoire avant une conversion plus stricte. Aucun quota
arbitraire de 95 % n'est imposé : les architectures diffèrent.

**Probabilité : inconnue, non estimable de façon fiable avant les pilotes.**
L'intervalle 10–30 % est retiré, faute de calibration. Ni le renommer, ni
répéter les essais ne justifie >90 %. Bonsai établit une faisabilité dans
cette famille de modèles ; il ne fournit pas la probabilité de cette campagne.

**Taille :** choix confirmé par l'utilisateur : privilégier Bonsai,
**environ 550–650 Mo pour le DiT**, plutôt que l'ancien plafond de 500 Mo.
Le payload calculé est de 550–614 Mo avant headers selon les groupes.
Viser un paquet complet ≤650 000 000 octets, métadonnées indispensables
comprises ; tout dépassement doit être exposé, pas compensé par une
ternarisation supplémentaire des supports. Text encoder et codec séparés.

## 1. Pourquoi changer le plan, sans oublier les échecs

Le [registre de preuves](TERNARY_V9_EVIDENCE_2026-09-25.md) est la référence
du diagnostic ; les chiffres ci-dessous ne sont pas des notes musicales.

| Observation vérifiée | Décision opérationnelle |
|---|---|
| V8 fait passer la velocity moyenne de 0,958326 à 0,847597 et le terminal de 0,769956 à 0,446067 sur la même sélection | Ne plus sélectionner une correction par sa seule erreur de reconstruction des poids |
| Le profil V8 consomme 256 états réels bruités, aucune trajectoire | Vérifier les quotas **consommés** de trajectoires teacher et student |
| Le meilleur record TTQ couvre sept matrices du bloc 0 | Ne pas extrapoler un bloc à un modèle complet ; tester entrée, milieu et sortie |
| TTQ peut avoir un centre et deux amplitudes asymétriques | Utiliser `W=s*q` pour le cœur ; les supports flottants ne justifient pas un faux alphabet ternaire |
| Cache, split, reprise et export ont des contrats différents | Un contrat commun et des tests négatifs avant une nouvelle optimisation |
| Un petit écart de rejeu dense et des symlinks avaient été surinterprétés | Distinguer défaut constaté, mécanisme plausible et cause démontrée |
| Une bonne similarité locale n'assure pas un bon audio | Débruitage complet et écoute dès le pilote, pas après 24 blocs |

Les versions précédentes avaient déjà laissé certains supports flottants :
leur maintien **ne résout donc pas à lui seul** les échecs. La nouveauté
essentielle est le protocole, la distribution de calibration et le choix
des checkpoints à partir des sorties réellement générées.

## 2. Ce que nous reprenons de Bonsai — et ce que nous ne savons pas

Bonsai Image part de FLUX.2 Klein 4B et conserve des projections sensibles
flottantes autour d'un cœur ternaire à groupes de 128. Son livre blanc
rapporte 94,4 % de rétention moyenne sur trois benchmarks ; cela n'impose
pas de reproduire exactement les pixels de la référence. Le déploiement
MLX est annoncé à 1,43 GB, distinct des 1,21 GB de représentation.
[Source primaire, §§4, 5, 7 et annexe D](https://github.com/PrismML-Eng/Bonsai-Image-Demo/blob/main/bonsai-image-4b-whitepaper.pdf).

Le [dépôt public consulté](https://github.com/PrismML-Eng/Bonsai-Image-Demo)
expose l'inférence. Les documents consultés ne donnent pas une recette
complète de conversion de SA3 : corpus, pertes, calendrier et budget
d'apprentissage restent à établir ici. Nous reproduisons **un périmètre et
un format**, pas une procédure d'entraînement secrète supposée connue.
L'appellation historique « Bonsai/Hadamard » ne prouve pas que cette
combinaison reproduisait la méthode de PrismML.

Deux sources guident les expériences, sans promettre leur transfert :

- [CAT-Q](https://arxiv.org/abs/2606.26650) : affectation des codes,
  transition douce/dure et fenêtres recouvrantes. Résultats sur LLM,
  pas sur ce modèle audio. Son [README officiel](https://github.com/IntelChina-AI/BitTern/blob/main/projects/cat-q/README.md)
  indique encore que le trainer sera publié séparément.
- [PTQ pour DiT audio](https://arxiv.org/abs/2510.00313) : calibration adaptée
  aux étapes de diffusion. Résultats sur Stable Audio Open en W8A8/W4A8,
  pas preuve de ternaire SA3.

Toutes les recettes chiffrées ci-dessous sont des **hypothèses de projet
à mesurer**, sauf mention explicite d'un résultat existant.

## 3. Périmètre final exact et stockage

### Cœur converti

Pour chacun des 24 blocs, convertir ces sept matrices :

```text
self_attn.to_qkv.weight
self_attn.to_out.weight
cross_attn.to_q.weight
cross_attn.to_kv.weight
cross_attn.to_out.weight
ff.ff.0.proj.weight
ff.ff.2.weight
```

Total revérifié dans les headers du checkpoint : **168 matrices,
1 358 954 496 éléments**, soit **93,5038 %** des 1 453 368 336 éléments
inventoriés. Ce dénominateur est celui des tableaux du checkpoint ;
l'inventaire séparera paramètres appris et buffers.

### Supports conservés

Conserver sans réentraînement initial ni réduction de précision :

- projections de conditionnement local `to_local_embed` des 24 blocs ;
- projections de conditionnement global, texte et temps, modulation ;
- projections d'entrée/sortie et convolutions de pré/post-traitement ;
- memory tokens, biais, normes, gates et embeddings auxiliaires ;
- projection apprise `cond.seconds_total_weight` et constantes temporelles.

Conserver les dtypes réels : majoritairement FP16, mais aussi le FP32
nécessaire au chemin temporel/conditionnement. Ne pas appliquer un cast
global sous prétexte que les supports Bonsai sont décrits comme FP16.

Le text encoder et le codec restent inchangés et **hors taille du DiT**.
Publier également la taille totale du pipeline. Les activations et les
accumulations gardent le contrat numérique natif.

### Format et comptabilité

Pour le cœur : `q ∈ {-1,0,+1}`, une scale FP16, `W=s*q`.
Zéro exact, niveaux symétriques. Aucun offset de groupe libre, résidu LoRA,
branche dense compensatrice ou dépendance au teacher à l'inférence.
Les biais vectoriels d'origine restent autorisés.

Headers lus sans charger les poids sur l'accélérateur :

| Groupe | Codes 2 bits | Scales FP16 | Supports natifs | Total hors conteneur |
|---:|---:|---:|---:|---:|
| 128 | 339 738 624 | 21 233 664 | 189 223 968 | **550 196 256 octets** |
| 64 | 339 738 624 | 42 467 328 | 189 223 968 | 571 429 920 octets |
| 32 | 339 738 624 | 84 934 656 | 189 223 968 | **613 897 248 octets** |

Commencer par **G128**, conforme au point de comparaison Bonsai.
Tester **G32** si G128 perd trop de qualité ; ne pas chercher d'abord
quelques Mo au prix d'une nouvelle fragilité. G64 reste un compromis
ultérieur, pas un axe de recherche supplémentaire au premier pilote.

Le calcul suppose **une seule scale stockée**. Si un kernel MLX attend
deux tableaux affines, dériver le second au chargement lorsqu'il est
déterministe et publier son coût mémoire ; ne pas le cacher dans les Mo.
Mesurer le fichier final, ses métadonnées et son padding. Ne pas assimiler
`log2(3)` à un packing réellement déployé à 1,58 bit.

## 4. P0 — réparer et prouver le banc de mesure

Avant optimisation, figer dans un contrat unique :

- hashes du DiT ARC, codec, text encoder, runtime, code et configuration ;
- versions Python/NumPy/MLX et exécutable réellement utilisable ;
- noms, shapes, dtypes et partition nominative `core/support/buffer` ;
- sélection des données, filiation, hashes, split et durée réelle ;
- prompts, seeds, sigmas FP32, bruit initial et bruits de réinjection ;
- source de chaque état et hash du checkpoint student qui l'a produit ;
- format de quantification, état complet de reprise et budget de ressources.

Réutiliser les composants existants après correction ; ne pas réécrire le
sampler ou créer un second validateur incompatible.

Tests de sortie obligatoires :

1. **Identité dense** : instrumentation et branche de transition à
   `lambda=0` reproduisent le teacher dans deux processus.
2. **Dtypes/caches** : sérialiser puis rejouer 16 états couvrant les sigmas
   et deux longueurs, sans cast systématique FP16.
3. **Dimensions** : vérifier l'axe tokens ; les 64 memory tokens ne sont
   pas soustraits de l'axe des 1536 canaux.
4. **Gradient/checkpointing** : mêmes gradients et update avec/sans
   recomputation ; aucune coupure accidentelle dans le suffixe gelé.
5. **Reprise** : sauver puis reprendre reproduit le prochain batch,
   RNG, moments, scheduler, codes et update, dans la tolérance déclarée.
6. **Export précoce** : un bloc puis deux blocs font un aller-retour
   codes/scales/supports exact, sans clé manquante ni quatrième code actif.
7. **Tests négatifs** : mauvais teacher, timestep FP16, latent modifié,
   split inversé ou fichier non déclaré doivent être refusés.
8. **Apprentissage synthétique** : sur un petit réseau dont la solution
   ternaire est connue, le quantificateur doit apprendre ; sinon corriger
   l'optimisation avant de conclure sur SA3.

Tolérances de parité d'un même calcul : codes/scales exacts ; forward
L2 relative ≤1e-3, cosinus ≥0,99999 par état, erreur absolue publiée.
Ce sont des seuils d'implémentation, **pas des critères musicaux**.
Le gradient surrogate est testé comme tel : la dérivée numérique du
quantificateur discontinu ne constitue pas sa référence.

Livrable : `measurement_contract_pass`. Sans lui, aucun pilote qualité.

## 5. P1 — référence audio et données adaptées à la génération

### Référence

Teacher = **DiT ARC original**, recette ping-pong native, mêmes bruits,
conditionnements et SAME-L pour toutes les comparaisons. BASE/Euler est
une autre expérience, pas un remplacement silencieux
([contexte SA3](https://arxiv.org/abs/2605.17991)).

Pas de LoRA de style dans la référence initiale. Si la cible devient un
modèle personnalisé, figer explicitement cette nouvelle référence et
recommencer son baseline avant conversion.

Rendre d'abord 8 canaries denses (4 prompts × 2 seeds), puis le dev.
Vérifier et écouter les rendus de référence : un défaut déjà présent
n'est pas automatiquement un défaut de quantification. Conserver aussi
les prompts difficiles ; ne pas retirer après coup ceux qui gênent.

### Séparation des données

- **Train** : gradients, affectation, sensibilité et collecte on-policy.
- **Dev** : choix de méthode, groupe et checkpoint. Les anciens audits
  réutilisés appartiennent ici, y compris ceux nommés « validation ».
- **Test scellé** : réservé à un seul candidat final figé.

Vérifier parents, sessions, doublons PCM et dérivés ; un symlink est permis
si sa cible et son rôle sont manifestés. Les 434/33/41 anciens exemples
doivent être réconciliés, pas certifiés indépendants sur leurs seuls noms.
Pas de remplissage par fichiers non déclarés.

Étude personnelle locale : conserver provenance et restrictions connues.
Aucun upload de corpus, achat, nouvelle collecte massive ou redistribution
automatique. Ajouter de nouvelles sources seulement pour une couverture
manquante identifiée.

### Cache pilote proposé

- 32 prompts train × 4 seeds × 8 étapes = **1024 états de trajectoire teacher**.
- 512 états de latents réels bruités, répartis sur les sigmas.
- Après le premier candidat : 512 états de ses propres trajectoires,
  avec le teacher réévalué **sur ces mêmes états student**.
- Dev distinct : 16 prompts × 2 seeds ; couverture de huit familles
  (rythmes, graves, piano/acoustique, ensembles, voix, ambiances,
  textures aiguës, changements de dynamique).

Échantillonnage proposé après collecte student : 50 % teacher,
25 % réel bruité, 25 % student. Journaliser les quotas effectivement lus.
Plusieurs timesteps ou seeds d'un prompt ne sont pas des exemples
statistiquement indépendants.

Crop latent 128 seulement pour le démarrage et la mesure mémoire.
Apprendre ensuite sur des états générés à la longueur cible : ne pas
supposer qu'un crop d'une longue trajectoire conserve exactement son
contexte. Mesurer la correspondance latent/durée décodée.

Le text-to-audio est obligatoire. Couvrir du conditionnement local non nul
et des sources tenues à l'écart avant toute revendication audio-to-audio
ou inpainting ; les supports flottants ne dispensent pas de ces tests.

## 6. P2 — pilote discriminant, sans nouvelle cascade aveugle

### 6.1 Comparaisons minimales

| Bras | Contenu | Question |
|---|---|---|
| D | Dense natif puis dense instrumenté | Le banc préserve-t-il la référence ? |
| B128 | Cœur ternaire G128, supports natifs figés | Le compromis Bonsai convient-il à SA3 ? |
| B32 | Même méthode, mêmes supports, groupes 32 | Une granularité plus fine récupère-t-elle la qualité ? |

Pas de bras « tout-matrices ternaire ». Un W4 poids seulement est permis
comme **diagnostic secondaire** si les deux bras ternaires échouent :
s'il réussit, le régime extrême devient un suspect ; cela ne prouve ni
l'impossibilité du ternaire ni la validité de toutes nos pertes.

Tester bloc 0, bloc 12 et bloc 23 **séparément**, puis une fenêtre [0,1].
Leur reste de réseau est dense au premier essai. Ces résultats ne valent
pas validation d'un modèle complet.

### 6.2 Une méthode principale, une extension de capacité

Point de départ : poids denses originaux. Ne pas réinitialiser les masters
à partir des codes arrondis V7/V8. Les anciens TTQ restent des témoins
historiques, pas des checkpoints compatibles avec le nouveau format.

**Premier bras d'apprentissage : calibration progressive inspirée de CAT-Q.**

Par groupe, apprendre `mu`, une scale positive `s` et un seuil positif
`delta`, les poids maîtres `M=W0` restant d'abord figés :

```text
u = (M - mu) / s
q = -1 si u < -delta ; 0 si |u| <= delta ; +1 si u > delta
W_hard = fp16(s) * q
```

`mu` modifie l'affectation, **jamais** le poids final par un `+mu`.
Initialiser `mu=0`, `delta=0.5`, puis la scale par une reconstruction
symétrique de moindre erreur ; documenter les groupes nuls, les égalités
aux seuils et les bornes évitant sous-flux/sur-flux FP16.

Utiliser un surrogate lisse borné puis un forward hard à gradient surrogate.
Une interpolation avec `W0` n'est autorisée que pendant l'entraînement :
identité à `lambda=0`, transition vers `lambda=1`, puis **au moins 20 %
des updates en hard pur**, avec l'arrondi FP16 réel des scales.
Toujours évaluer séparément le hard, même durant la phase douce.
Fixer l'implémentation et ses tests avant le premier run ; ce port n'est
pas présenté comme une reproduction du trainer officiel non publié.

**Extension conditionnelle : QAT des masters de la fenêtre active.**

Si la calibration ne réduit plus l'erreur **train** représentative alors
que le banc synthétique et les gradients sont corrects, libérer les masters
denses dans 1–2 blocs seulement, avec le même quantificateur et la même loss.
Cela teste la capacité d'adaptation des poids, sans lancer Adam sur le
modèle entier. Ni davantage de corpus ni une grille de seuils ne remplace
ce test si le modèle ne sait déjà pas ajuster le train.

Les supports restent identiques à l'original dans les deux bras.
Une adaptation future de supports serait une ablation distincte, non le
mécanisme implicite chargé de compenser un cœur défaillant.

### 6.3 Apprendre la fonction du DiT, pas seulement les poids

Cible principale : `T(x, sigma, c)` du teacher figé.
Ne pas la remplacer par `epsilon - latent_reel` d'un autre entraînement.

```text
Nv = ||S(x)-T(x)||² / max(||T(x)||², epsilon)
Nw = erreur normalisée de sortie de fenêtre sur une entrée commune
L_initial = Nv + 0.25*Nw
```

Le coefficient et le plancher sont des valeurs initiales à configurer.
La velocity se mesure à la sortie du **DiT complet**, gradient traversant
le suffixe gelé. La loss locale ne décide jamais seule de la promotion.
Publier les erreurs par sigma, famille, source d'état et les dérives de niveau.

Le préfixe student doit fournir les entrées réellement rencontrées.
Comparer les fonctions teacher/student d'une fenêtre sur la même entrée,
tout en conservant la cible globale `T(x)`. Invalider le cache de fenêtre
quand le préfixe change.

Rafraîchir les trajectoires student à chaque checkpoint retenu et au plus
toutes les 100 updates ; leur hash fait partie du cache. Générer puis
étiqueter ces états dans des processus séparés. Pas de gradient sur
le teacher, pas d'utilisation du test dans la collecte on-policy.

Si les prédictions sur états identiques progressent mais les générations
dérivent : une ablation ajoute une loss de transition ping-pong sur deux
pas, mêmes bruits sauvegardés. Elle utilise la vraie transition, pas une
approximation Euler ; son coût mémoire doit être mesuré avant adoption.
Ne pas imposer immédiatement une différentiation sur huit pas et le codec.

### 6.4 Budget et décision du pilote

Valeurs initiales proposées, pas résultats acquis :

- batch 1, accumulation 4, clipping norme 1, recomputation testée ;
- paramètres de quantification : LR 3e-4 ; unique réduction à 1e-4
  si les diagnostics montrent de l'instabilité ;
- masters éventuels : LR 1e-5 séparé, départ depuis le meilleur état
  continu sauvegardé, pas depuis les codes ;
- 500 updates maximum au premier essai, audit hard toutes les 50 ;
- extension à 1500 seulement si le dev et les rendus progressent ;
- arrêt/diagnostic après quatre évaluations sans progrès du hard ;
  NaN, échec de contrat ou dépassement mémoire arrêtent immédiatement ;
- refaire le bras sélectionné sur un deuxième seed d'optimisation.

Avant généralisation, les pilotes d'entrée/milieu/sortie et la fenêtre
[0,1] doivent passer les contrôles techniques et une écoute A/B courte :
pas de nouvelle statique, disparition de contenu, transitoires détruites
ou timbre manifestement dégradé. Archiver les fiches, pas seulement « OK ».

Les anciens seuils de cosinus 0,95/0,90 sont désormais des **repères
diagnostiques**, pas des critères de fidélité musicale. Ce changement est
fait **avant tout entraînement V9**, pas pour reclasser V8 comme réussi.
Un écart numérique important impose inspection et rendu, pas une
déclaration automatique d'échec ou de qualité.

Sortie : `pilot_accepted`, avec bras/groupe/recette/hashes et deuxième seed.
Sans écoute : `musical_review_pending`, aucune cascade longue.

## 7. P3 — convertir le cœur progressivement et conserver la qualité

Après le pilote uniquement :

1. Convertir les fenêtres [0,1], [1,2], …, [22,23].
2. Garder le préfixe déjà converti dans le vrai forward.
3. Réouvrir le bloc partagé ; ne pas figer définitivement sa calibration
   sur la distribution du teacher dense.
4. Faire un canary hard après chaque fenêtre ; audits complets et écoutes
   aux jalons 2, 4, 8, 12, 18 et 24 blocs.
5. Garder le meilleur checkpoint global, pas systématiquement le dernier.
6. À la première dégradation confirmée, restaurer le dernier jalon accepté
   et réoptimiser la dernière fenêtre ; une seule réouverture supplémentaire
   de la fenêtre précédente est permise par incident.
7. À couverture complète, une passe de récupération recouvrante maximum ;
   ajouter une autre passe seulement avec une hypothèse et un budget revus.

Les masters continus utiles à une réouverture sont conservés sur disque/CPU
avec leur filiation. Seule la fenêtre active et ses moments résident sur
l'accélérateur. Une reprise interrompue restaure l'état exact ; un nouvel
essai qui réinitialise l'optimiseur reçoit un nouvel identifiant.

Une matrice non convertie ne peut pas être cachée dans « supports » :
l'allowlist finale est celle de la section 3. Inversement, aucun lot ne
prévoit de convertir ultérieurement les supports déclarés.

## 8. P4 — export autonome et ressources mesurées

### Export

Exporter tôt puis après chaque jalon significatif :

- les 168 matrices du cœur en codes/scales, avec couverture 168/168 ;
- les 357 autres tableaux du checkpoint à leur dtype natif, conservés
  bit à bit tant qu'ils n'ont pas fait l'objet d'une ablation autorisée ;
- noms, shapes, convention de padding, révision du chargeur et hashes ;
- taille du DiT séparée de celle des composants externes.

Le processus de vérification ne reçoit **aucun checkpoint dense du DiT**.
Le paquet doit contenir ses supports, pas les récupérer depuis le teacher.
Contrôler les fichiers effectivement ouverts ; les composants texte/codec
déclarés restent autorisés.

Tester égalité codes/scales et parité avant/après export ; puis refaire les
rendus avec **l'artefact rechargé**, pas avec l'objet d'entraînement.
Le packing ne garantit ni kernel ternaire accéléré ni faible pic mémoire :
publier la latence et le pic, y compris une éventuelle déquantification.

### Ressources

Conserver le plafond local de **12 000 000 000 octets Metal** pour le travail
mesuré, avec arrêt préventif à 11 000 000 000. Rapporter octets et GiB, RSS,
pression système et variation de swap ; ne pas additionner aveuglément CPU
et GPU sur mémoire unifiée.

- Teacher et student dans des processus successifs, cibles shardées.
- Text encoder déchargé après calcul des conditions ; codec seulement pour
  les rendus/audits. Supports du DiT toujours présents à leur dtype natif.
- Shards de 128–256 états ; jamais tous les caches de 24 blocs simultanément.
- Un essai à la fois ; mesurer initialisation, forward, backward, update,
  sauvegarde, reload et génération à durée cible.
- Hashes binaires streamés ; pas de `q.tolist()` géant.
- Checkpoints atomiques et reprise ; préserver les résultats historiques.
  Aucune suppression de corpus/anciens essais pour libérer de l'espace.

Disque : mesurer un shard, une fenêtre avec Adam et un export, puis calculer :

```text
libre_requis = réserve_système_10_GB
             + 1.25 * nouveaux_octets_max_simultanés_prévus
```

Inclure best/reprise, masters conservés, sauvegarde temporaire atomique,
caches et WAV ; déduire seulement les fichiers déjà présents et réutilisables.
Ne pas recopier le teacher dans chaque run. Pas de seuil universel 40/60 GB
présenté comme une nécessité scientifique.

Au contrôle de cette révision : **7 555 288 KiB libres**, soit environ
7,74 GB / 7,21 GiB. La réserve de 10 GB n'est pas atteinte : les tâches
documentaires restent possibles, l'entraînement attend une nouvelle
vérification d'espace ou un emplacement de travail approprié.

L'ancien `.venv` du runtime est absent ; Python local 3.12.6/MLX 0.31.2
était utilisable lors de l'audit. Verrouiller et retester l'environnement,
sans réinstallation aveugle ni supposition de disponibilité future.

Enveloppes proposées : 12 h de calcul pour le pilote, 96 h pour la campagne.
Ce ne sont ni des prévisions de durée ni des plafonds imposés par Bonsai.
Avant extension, mesurer le coût des 100 premières updates, des audits et
rendus, extrapoler avec marge ×2 et publier le budget. Aucun GPU payant,
service externe ou téléchargement massif n'est implicitement autorisé.

## 9. P5 — démontrer la qualité, puis décider

### Trois verdicts séparés

| Verdict | Condition |
|---|---|
| `format_pass` | Cœur 168/168 ternaire symétrique, supports déclarés, export autonome, octets mesurés |
| `technical_pass` | Contrat/reload corrects ; audio fini, durée/canaux corrects ; aucun nouvel échec technique |
| `quality_accepted` | Comparaison audio aveugle et appariée acceptée sur le test réservé |

Un fichier qui charge ou une loss qui baisse ne peut pas produire
`quality_accepted`. Un minimum de similarité latente ne peut pas le
remplacer non plus. Les matrices de support flottantes **n'empêchent pas**
la réussite du format demandé.

### Test réservé, préparé avant optimisation

- 40 unités de tâche indépendantes réparties sur huit familles,
  deux seeds par unité ; regrouper dérivés/parents/prompts quasi identiques.
- Définir les durées avant test : cas courts, rendus 30 s et au moins
  six paires 180 s sur une sous-sélection fixée d'avance.
- Les usages audio-to-audio/inpainting ont leurs propres cas et leur
  propre verdict ; ne pas revendiquer un mode non évalué.
- Choisir un seul artefact sur dev, le hasher, puis ouvrir le test.
  Un test utilisé pour corriger devient dev ; nouveau test pour conclure.

Conserver les WAV float32 avant garde de niveau. Mesurer finitude,
silence/statique, durée, RMS/LUFS, DC, clipping, true peak, spectre,
transitoires et continuité par fenêtres. Ne pas appeler « clipping »
tout pic float >1 : distinguer overshoot récupérable et écrêtage effectif.

Écoute aveugle A/B à niveau comparable : définition, grave, aigus,
transitoires, stéréo, continuité, respect du prompt et utilité musicale.
Autoriser seulement un gain constant documenté pour l'écoute/livraison ;
aucun EQ, compresseur, débruiteur, de-esser ou limiteur réparateur.

Critère proposé à figer dans P0 : différence moyenne de qualité
student−teacher sur 100, borne inférieure unilatérale à 95 % **>−5 points**,
sans nouveau défaut rédhibitoire. Évaluer séparément le respect du prompt.
Publier les résultats par famille ; la moyenne ne masque pas une famille
qui échoue.

Calculer l'incertitude par unités indépendantes, avec bootstrap stratifié
par famille et regroupement des deux seeds d'une unité. Ne pas traiter les
pas de diffusion, les secondes audio ou les notes d'un même extrait comme
des observations indépendantes. Avec un seul auditeur, la conclusion reste
conditionnelle à cet auditeur et à ce jeu de tâches.

Si l'intervalle est trop large : `inconclusive`. Prévoir avant l'ouverture
une extension de test à taille fixe, ou reporter une nouvelle campagne ;
ne pas ajouter des exemples jusqu'à obtenir le chiffre souhaité.
Sans écoute humaine : `musical_review_pending`.

Une rétention de benchmark ou une fiabilité après livraison n'est pas
la probabilité que la recherche réussisse. Aucun pourcentage de succès
du plan n'est annoncé sans fondement.

## 10. Boucle bornée : un échec doit départager une hypothèse

| Résultat | Prochaine action autorisée dans le plan |
|---|---|
| Dense instrumenté différent / export divergent | Corriger le banc, invalider le run ; pas de tuning qualité |
| Train ne s'ajuste pas malgré gradients valides | Tester les masters entraînables dans la fenêtre |
| Train progresse, dev se dégrade | Vérifier couverture et surapprentissage ; ajouter des données ciblées |
| Soft bon, hard mauvais | Corriger transition, surrogate, arrondi des scales ; pas de livraison soft |
| Bloc bon, génération mauvaise | Entrées student, suffixe, fenêtre recouvrante et éventuellement deux pas |
| B128 rejeté, B32 accepté | Retenir B32 et sa taille réelle ; ne pas convertir les supports pour compenser |
| Les deux ternaires échouent, W4 passe | Isoler le quantificateur/capacité ; W4 reste un témoin, pas un succès ternaire |
| Qualité bonne mais taille > enveloppe de 650 Mo | Vérifier packing/métadonnées puis arbitrer, sans convertir les supports en cachette |
| Technique bonne, écoute mauvaise | Rejet qualité ; ne pas masquer les défauts par du mastering |

Maximum trois rounds de pilote : protocole/couverture, capacité du
quantificateur, puis interaction de fenêtre selon la cause observée.
Chaque round doit apporter un changement motivé et un témoin apparié.
Les essais ne sont pas supposés indépendants.

Si aucun pilote n'est accepté : publier `conversion_not_demonstrated`,
la meilleure preuve et la cause encore incertaine. Un autre modèle,
un entraînement natif ou davantage de calcul sont de nouvelles options
à discuter, pas une phase automatique ni un constat d'impossibilité.

## 11. Lots d'implémentation et première action

Réutiliser les outils du dépôt après validation de leurs contrats.
Le tableau reste le découpage de campagne ; l'état courant est ajouté pour
éviter de confondre une étape conçue avec une étape réellement démontrée.

| Ordre | Travail | Critère de sortie | État au 25/09 |
|---|---|---|---|
| P0 | Unifier `ternary_runtime_contract.py`, provenance V8, audit/reprise/export ; tests d'identité et synthétiques | Banc fiable et erreurs injectées refusées | **PASS** |
| P1 | Étendre `build_ternary_state_cache.py` et les sélections ; produire baseline dense et dev représentatif | Couverture réelle, baseline écoutée | **PASS technique** ; écoute pending |
| P2 | Pilote full-DiT B128/B32, puis on-policy conditionnel | Pilote rechargé accepté, deux seeds | **G32 PASS technique** sur [0,1] deux seeds ; bloc 2 PASS on-policy ; écoute pending |
| P3 | Scheduling des fenêtres recouvrantes et cascade cumulative | Jalons acceptés, supports inchangés | **SUSPENDU** jusqu'à A/B et gel recette |
| P4 | Export/compactage/vérification autonome Bonsai | 168 matrices, supports embarqués, aucun teacher requis | Non lancé pour le modèle complet |
| P5 | Rendus, fiches A/B et test réservé | Format, technique et qualité acceptés séparément | Canaris techniques ; test réservé/A-B non faits |

Chaque run garde : contrat et digests, configuration résolue, logs d'updates,
évaluations hard, couverture consommée, ressources, checkpoint de reprise,
artefact, WAV bruts, fiches d'écoute et décision motivée.
Reporter les conclusions dans la
[base de connaissances](TERNARY_DISTILLATION_KNOWLEDGE_BASE.md).

**Première action à l'exécution : P0, puis 8 canaries denses et le pilote
B128 ; B32 seulement si nécessaire. Pas de relance immédiate des 24 blocs.**

Le livrable recherché est désormais un **DiT au compromis Bonsai**.
Aucune étape finale ne ternarise les supports et aucune promesse de
réussite >90 % ne remplace les preuves.

## 12. État d'exécution V9.1 — 25 septembre 2026

Cette section remplace le statut documentaire initial. Les détails et les
limites sont dans le [rapport d'exécution](TERNARY_V9_EXECUTION_REPORT_2026-09-25.md)
et le [registre de preuves](TERNARY_V9_EVIDENCE_2026-09-25.md).

### P0/P1 réellement exécutés

- Le contrat header-only du teacher réel couvre **168 matrices cœur** et
  **357 supports natifs**, soit 93,5038 % des éléments. Le contrat courant est
  [`contract-v9.1-final.json`](../output/sample-expertise-pilot/ternary-quality-v9-bonsai-p0/contract-v9.1-final.json).
- Les enveloppes hors conteneur mesurées par le contrat sont **550 196 256
  octets en G128** et **613 897 248 octets en G32**, toutes deux dans la
  limite de 650 000 000 octets avant métadonnées supplémentaires.
- Parité dense en deux processus, replay du cache sur 16 états, reprise/RNG,
  gradient-checkpointing et tests de packing passent. Les caches réels
  contiennent 512 états et 512 cibles teacher, avec filiation et hashes.

### P2 : résultat du pilote

Le bras retenu est **G32**, avec supports inchangés et quantification
symétrique `W=s*q`. Les matrices et les records sont rechargés depuis les
checkpoints ; aucune grosse exportation dense de 2,6 Go n'est utilisée pour
les nouveaux runs.

| Expérience | Velocity mean/min | Gate release | Audio canary apparié |
|---|---:|---|---:|
| G128 [0,1], 112 updates | 0,96056 / 0,86240 | passe | cos 0,96927 ; L2 0,2524 |
| G128 [1,2], 112 updates | 0,95200 / 0,82577 | **échoue** | cos 0,95517 ; L2 0,3014 |
| G32 [0,1], seed 20260925 | 0,96492 / 0,88516 | passe | cos 0,96555 ; L2 0,2671 |
| G32 [0,1], seed 20260926 | 0,96193 / 0,86794 | passe | cos 0,93518 ; L2 0,3762 |
| G32 bloc 2, pointwise | 0,95662 / 0,84431 | **échoue** | — |
| G32 bloc 2, on-policy 4 pas | 0,95735 / 0,85615 | passe | cos 0,94665 ; L2 0,3241 |

Le résultat montre pourquoi la loss locale ne suffit pas : G32 [0,1] passe,
mais le bloc 2 ne passe qu'après exposition aux états de trajectoire student.
Le rollout on-policy 4 pas reste sous la garde de 11 Go, avec un pic observé
de 7,85 Go. Une tentative G128 sur deux blocs a dépassé la garde à 12,33 Go
et a été arrêtée ; elle ne doit pas être relancée telle quelle.

### Décision et prochain gate

Le pilote technique est **prometteur mais pas encore `quality_accepted`**.
Les WAV bruts sont finaux techniquement, mais aucune écoute A/B humaine
aveugle n'a encore validé l'équivalence musicale. Les canaris disponibles
dans le rapport servent précisément à cette revue ; ils ne remplacent pas
le jugement musical.

En conséquence, **P3 (cascade des 24 blocs) reste volontairement suspendu**
jusqu'à la revue audio et au gel de la recette G32. Aucun modèle complet
168/168, export autonome Bonsai final ou test réservé n'est déclaré livré.
