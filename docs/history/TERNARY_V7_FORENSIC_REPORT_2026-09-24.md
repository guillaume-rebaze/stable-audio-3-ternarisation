# Audit V7 — pourquoi les essais ternaires n'ont pas abouti

Date : 24 septembre 2026. Périmètre : Stable Audio 3 Medium / MLX.
Statut : photographie de l'investigation préalable aux runs P2/P3. Depuis,
le cache/RNG corrigé, P2 et l'ablation P3 ont été exécutés ; leurs mesures et
la décision courante sont consignées dans le
[rapport d'exécution V7](TERNARY_V7_EXECUTION_REPORT_2026-09-24.md).

Plan actif : [V7](TERNARY_QUALITY_RECOVERY_PLAN.md).
Historique : [V6 figée](archive/TERNARY_QUALITY_RECOVERY_PLAN_v6_2026-09-24.md),
[base de connaissances](TERNARY_DISTILLATION_KNOWLEDGE_BASE.md).

## 1. Conclusion

Les essais ne démontrent pas que la ternarisation de SA3 est impossible.
Ils montrent un échec de fidélité, avec trois défauts logiciels désormais
reproduits et une optimisation beaucoup moins effective que supposé :

1. Les cinq raffinements successifs des blocs 0–3 se terminent avec
   **exactement les mêmes 226 492 416 codes ternaires**. Les scales changent.
   Les poids maîtres et l'optimiseur sont perdus entre les passes.
2. Le teacher utilisé pour les cibles reçoit des timesteps FP16, alors que
   la génération emploie des timesteps FP32. La différence seule provoque
   jusqu'à **9,55 % d'erreur relative de velocity teacher** sur notre sonde.
3. Les biais vectoriels FFN sont entraînables, mais absents des sauvegardes
   records-only. La matérialisation remet ceux du teacher. Le reload exact
   ne compare pas nécessairement le modèle entraîné au modèle livré.

Ces défauts doivent être corrigés avant de conclure sur la capacité du
quantizer, le volume du corpus ou une nouvelle loss. Aucun n'est démontré
comme cause unique de tout l'écart audio. Les nouveaux tests ne sont pas
une mesure de réussite de V7.

## 2. Ce qui a été examiné

- Plans et investigations initiaux, registre des preuves, rapports V3/V4,
  proposition V5, plan et comptes rendus V6.
- Code de quantification, checkpointing, réouverture des fenêtres, calcul
  des cibles, export, audit, sampler ARC pingpong et rendu SAME-L.
- Six snapshots successifs des mêmes 28 projections, rapports de validation,
  manifests de données, artefact final partiel.
- Nouvelles sondes bornées : comparaison des codes sur CPU, 24 états TRAIN
  pour isoler la précision temporelle, roundtrip synthétique du biais,
  trois paires audio teacher/student sur validation déjà consultée.
- Recherche primaire revue au 24 septembre, synthèse et décisions dans le
  plan V7. Aucun article ne valide exactement notre ensemble de contraintes.

L'index de code local antérieur à ces scripts ne couvrait pas le sujet :
les conclusions viennent de la lecture directe et des mesures, pas du graphe.

Au moment de cette investigation, aucun entraînement long n'avait été lancé ;
aucun checkpoint V6 n'avait été modifié et aucun test scellé n'avait été
ouvert. Les probes décrites ici sont des diagnostics directs du modèle, pas
un E2E du service MusicGen ou de l'application. Les runs ultérieurs sont
hors du périmètre temporel de cette photographie et figurent dans le rapport
d'exécution lié ci-dessus.

## 3. Chronologie des échecs

Les chiffres ci-dessous viennent des rapports historiques cités ; ils
n'ont pas tous été remesurés ce jour. Comparer des lignes ayant des datasets
différents ne constitue pas une ablation contrôlée.

| Étape | Observation conservée | Limite ou erreur déterminante |
| --- | --- | --- |
| Bonsai/cascade initiale | ~493,8 Mo ; cosinus de blocs >0,99, velocity complète ~0,5596, minimum ~0,1735 | Bon bloc local ≠ bonne fonction cumulée ; scopes de quantification, rotation et rendu non alignés |
| Récupération affine G64/G32 | ~512,63 / 580,27 Mo ; erreurs relatives reload ~0,215 / 0,249 % | Moyenne affine non conforme au ternaire symétrique ; buffers temporels castés ; tests incomplets et données déjà vues |
| QAT globale initiale | Pic Metal 29,42 GiB | Implémentation globale naïve incompatible avec le budget ; pas preuve d'impossibilité de toute QAT fenêtrée |
| V3 stricte G64 | 479 660 202 octets ; velocity ~0,8304 / 0,7184 ; deux rendus sur trois hors gate RMS | Cascade locale, couverture limitée ; défaut Adam FP16 corrigé, fidélité audio non restaurée |
| V4 autorisée | 479 660 087 octets ; validation 0,81248 / 0,65934 ; trois rendus sur trois hors gate RMS | Config déclarée différente de l'exécutée ; identité de source approchée ; ancien test consulté |
| V5 | Proposition d'étudiant natif/global | Pas un succès exécuté ; budget mémoire de l'optimiseur sous-estimé |
| V6 | Réparation du gradient checkpointing, fenêtres, caches et raffinements ; seuls blocs 0–3 convertis | Reprises sans état maître ; précision temporelle incohérente ; biais perdus ; gates partiels trop faciles à confondre avec une livraison |

Sources : [registre](TERNARY_RECOVERY_EVIDENCE_2026-09-22.md),
[rapport V3](TERNARY_V3_EXECUTION_REPORT_2026-09-22.md),
[rapport V4](TERNARY_V4_AUTHORIZED_EXECUTION_REPORT_2026-09-22.md),
[V5](TERNARY_V5_NATIVE_STUDENT_PLAN.md), [V6](archive/TERNARY_QUALITY_RECOVERY_PLAN_v6_2026-09-24.md).

### Les gains V6 existaient, mais ne suffisaient pas

Même validation SFT Voices, 12 prompts, seed de base 4242 :

| Variante V6 | Velocity ponctuelle moyen / min | Cosinus latent terminal moyen / min |
| --- | ---: | ---: |
| Cache 2048, contrôle antérieur | 0,93584 / 0,80593 | 0,61383 / 0,45651 |
| Corpus SFT, états teacher | 0,94239 / 0,84282 | 0,64923 / 0,51250 |
| États student, 8 pas | 0,94431 / 0,79815 | 0,66628 / 0,57049 |
| États student, 16 pas, pondération sigma | 0,94813 / 0,87302 | 0,66579 / 0,54982 |

Le dernier gate ponctuel « release » passe ; le terminal plafonne.
Le dernier fichier pèse **2 324 603 402 octets**, avec 20 blocs encore denses.
Les 455,8 Mo ne sont qu'une estimation du futur scope intégral.

La table documente le protocole historique, qui contient le décalage de
timesteps décrit ci-dessous. Ne pas comparer ses valeurs directement à un
futur audit V7 corrigé sans réévaluer le contrôle V6.

## 4. Preuve A — les reprises n'ont pas réorganisé les codes sauvegardés

Données : [forensics.json](../output/sample-expertise-pilot/ternary-quality-v7-forensics-20260924/forensics.json).
Les chemins complets et empreintes de chaque snapshot y sont conservés.

| Transition, toujours sur 28 matrices G32 | Codes différents | Scales différentes |
| --- | ---: | ---: |
| Cascade 0–3 → première réouverture conjointe | 0 / 226 492 416 | 85,64 % |
| Réouverture → cache étendu 2048 | 0 / 226 492 416 | 86,33 % |
| Cache étendu → SFT teacher | 0 / 226 492 416 | 86,86 % |
| SFT teacher → SFT student 8 pas | 0 / 226 492 416 | 86,11 % |
| SFT student 8 → 16 pas pondérés | 0 / 226 492 416 | 85,76 % |

Ce sont cinq passes de 250 updates, soit 1 250 updates et environ 2 h 35
d'entraînement rapporté, hors construction des caches et audits.

Mécanisme lu dans [train_ternary_window_v6.py](../services/musicgen/train_ternary_window_v6.py) :

- quantized_linear_to_qat reconstruit le maître FP32 à partir des codes/scales
  arrondis du précédent artefact.
- AdamW est recréé ; les moments et la position du scheduler sont perdus.
- Les records finaux ne sauvegardent ni les maîtres ni les états optimiseur.
- Le mode symmetric recalcule ses scales à partir des poids ; ce ne sont pas
  des scales indépendantes apprises comme dans le mode learned_symmetric.

L'initialisation depuis un poids déjà projeté efface les déplacements
continus qui n'avaient pas encore franchi un seuil. Des gradients non nuls
et une loss qui bouge ne prouvent donc pas une évolution des codes durs.

**Limite :** égalité des snapshots = aucun changement net. Des flips
intermédiaires suivis d'un retour sont possibles ; ils n'étaient pas journalisés.
Le lien causal entre reset, LR, distances aux seuils et plateau doit être
séparé par une ablation. « La quantification ne peut pas apprendre » n'est
pas une conclusion justifiée.

## 5. Preuve B — deux contrats de timestep incompatibles

La génération ARC pingpong compose un scalaire FP32 avec le batch ; dans
le runtime MLX installé, le tenseur t résultant est FP32. En revanche :

- [prepare_ternary_teacher_targets.py](../services/musicgen/prepare_ternary_teacher_targets.py)
  convertit sigma en FP16 avant le forward teacher.
- Le trainer V6 et l'audit ponctuel font le même arrondi.
- Dans l'audit teacher-on-student, la velocity student de la trajectoire
  et celle du teacher recalculé n'utilisent pas le même dtype de t.

Exemple : 0,9943755865 devient 0,994140625 en FP16. Le maintien des fréquences
Fourier en FP32, déjà corrigé précédemment, ne répare pas cet arrondi d'entrée.

### Sonde isolée, 24 états TRAIN

Trois prompts du cache de train, huit états de trajectoire teacher par prompt.
Même x, même conditionnement, mêmes poids ; seule la précision de t change.

- Erreur relative teacher seule : moyenne **1,183 %**, maximum **9,548 %**.
- Cosinus teacher seul : moyenne 0,999716, minimum 0,995433.
- Au pire point de précision, écart relatif des features Fourier : 81,08 %.
  Ce nombre n'est ni l'erreur velocity ni une erreur audio.
- Student contre teacher avec t FP32 : cosinus moyen **0,94554**, minimum
  **0,72723** ; erreur relative moyenne **0,29845**.
- Le pire point student est le prompt funk à **sigma=1**, où FP16 et FP32
  représentent exactement le même timestep : le défaut temporel n'explique
  donc pas à lui seul la dégradation.
- À ce point, cosinus de la prédiction débruitée x−sigma*v : **0,68068**.

La grille ponctuelle historique [0,95 ; 0,75 ; 0,5 ; 0,25 ; 0,1] ne couvre
ni sigma=1 ni exactement les timesteps du sampler employé.

Ces 24 points isolent un bug ; ils ne mesurent pas une généralisation.
Temps de cette exécution : 50,98 s ; pic Metal : 6,081 GiB.

### Pourquoi « l'erreur apparaît tard » ne suffit pas

Dans pingpong, schématiquement, avec bruit partagé :

~~~text
d = x - sigma * v
x_next = (1 - sigma_next) * d + sigma_next * epsilon
Delta_x_next = (1 - sigma_next) * (Delta_x - sigma * Delta_v)
~~~

Un fort bruit commun peut masquer un désaccord de débruitage aux premiers
pas. La divergence visible tard ne localise pas forcément la cause tard.
Une loss sur les deux derniers pas peut être utile, mais ne remplace ni le
correctif de précision ni la couverture des premiers pas.

## 6. Preuve C — le format records-only perd les biais appris

Dans chaque FFN ouverte, ff.ff.0.proj et ff.ff.2 ont des biais vectoriels.
module.unfreeze() les rend entraînables avec les poids : **55 296 scalaires**
sur quatre blocs.

save_records_checkpoint conserve packed_codes, scales et biases. Attention :
**biases désigne le décalage technique du quantizer MLX**, pas le paramètre
bias de nn.Linear. L'export repart du teacher dense et récupère ses biais.

La [sonde de biais](../output/sample-expertise-pilot/ternary-quality-v7-forensics-20260924/bias-forensics.json)
reproduit exactement ce chemin de sauvegarde/rechargement :

- petite couche de contrôle, entrée nulle ;
- modification synthétique du biais de 0 à 0,125 ;
- sortie avant sauvegarde : [0,125 ; 0,125] ;
- sortie après records-only + matérialisation : [0 ; 0].

Sur le dernier export réel, les huit tenseurs de biais FFN des blocs 0–3
sont exactement ceux du teacher. L'amplitude des mises à jour de biais
effectivement réalisées pendant les entraînements n'est plus récupérable :
ne pas lui attribuer un pourcentage d'échec inventé.

Un reload exact d'un modèle déjà rematérialisé vérifie seulement la seconde
moitié du trajet. Il faut comparer **dernier forward QAT dur → fichier →
nouveau processus**, avec tous les paramètres entraînés.

## 7. Ce que les rendus bruts montrent réellement

[Mesures audio](../output/sample-expertise-pilot/ternary-quality-v7-forensics-20260924/audio-v6-control/audio_metrics.json)
et six WAV conservés à côté. Même teacher ARC, même SAME-L, mêmes graines,
huit pas ; validation déjà utilisée. Aucun gain, EQ, limiteur ou clipping ajouté.

| Prompt de validation | Seed | RMS student / teacher | Corrélation spectrale mid | Corrélation enveloppe mid |
| --- | ---: | ---: | ---: | ---: |
| Voice 112 BPM | 4242 | 0,818 | 0,409 | 0,869 |
| Voice 136 BPM | 4243 | 0,897 | 0,193 | 0,414 |
| Voice live 99 BPM | 4244 | 1,091 | 0,328 | 0,740 |

Les six fichiers sont finis, stéréo 44,1 kHz, sans dépassement de 1 en
sample peak. Durée décodée **11,8886 s**, pour crop_len=128 et condition 12 s.
Ce n'est pas un test à 30 ou 180 secondes. Pic Metal : 8,432 GiB.

Les champs candidate_pass=true du renderer actuel sont égaux au gate
technique amplitude/canaux/valeurs finies. Ils **ne signifient pas**
« musicalement accepté ». Les corrélations modestes ne sont pas non plus
une preuve universelle de mauvais son : un arrangement peut différer.

Statut d'interprétation : **diagnostic_only / musical_review_pending**.
Aucune écoute humaine effectuée dans cet audit. Il serait inexact de
présenter ces trois sorties comme un collapse sonore établi ou comme une
réussite musicale.

## 8. Données, contraintes et inconnues

- Corpus V6 : train 434 latents / 383 parents nommés / 85 prompts ;
  validation 33 latents / 12 prompts ; test 41 latents / 24 prompts.
- Le test est resté fermé. Les splits par nom, prompt et hash latent ne
  démontrent pas l'absence de doublons d'œuvre, de session ou de PCM.
- Une validation surtout vocale ne couvre pas toutes les musiques.
  Ajouter une validation de développement diversifiée, pas recycler le test.
- 250 updates avec accumulation 4 = 1 000 tirages avec remise, pas
  nécessairement une passe complète sur 2 048 ou 2 720 états.
- Les résultats ne permettent pas d'isoler l'effet du corpus, des scales
  apprises ou de la durée d'entraînement : des défauts communs subsistent.
- Le mode learned_symmetric essayé possède un surrogate à sharpness fixe.
  Cela n'est pas la transition douce-vers-dure planifiée en V6.
- La mémoire 12 GB signifie 12 000 000 000 octets, pas 12 GiB.
  Le dernier QAT fenêtré était à ~8,74 GiB, pas le modèle entier entraîné.
- Espace observé ce jour : interne ~51 GiB libres ; Extreme SSD ~7,9 GiB.
  Ces valeurs sont temporaires et doivent être remesurées avant exécution.

Le degré de récupération possible après correction, la durée d'optimisation
requise et la qualité d'un scope entièrement ternaire restent inconnus.
Ce sont les questions expérimentales de V7, pas des bugs déjà résolus.

## 9. Reproduire les diagnostics

Depuis la racine du dépôt, environnement MLX existant. Les sorties doivent
être nouvelles ; le script refuse d'écraser un rapport existant.

~~~sh
rtk proxy python services/musicgen/audit_ternary_v7_forensics.py \
  --probe-timesteps --probe-biases \
  --output output/sample-expertise-pilot/ternary-quality-v7-forensics-recheck/report.json
rtk proxy python -m pytest -q services/musicgen/tests/test_ternary_v7_forensics.py
~~~

Ne pas ouvrir le test scellé pour ces vérifications. La sonde de t lit le
cache TRAIN V6 ; la sonde de biais crée une petite fixture, sans modifier
les poids historiques.

Empreintes des entrées principales :

~~~text
teacher dit_medium_f16.npz
f9e5647ea3225818657d47d47ae4b34afa29c0568206ca89566c1a758944a38e

student core_g32_blocks0-3_sftvoices_student16.npz
de20f28cd4425fa879a092aab268a50435311818368256f94ceba0053f123919

runtime sa3_pipeline.py
e8aca5225eeda40e1d3f53c04968bbce9df6290021f4bdc5c1a67d6d68013de8
~~~

Le rapport de t et celui des biais enregistrent chacun l'empreinte du script
au moment de leur exécution ; la sonde de biais a été ajoutée entre les deux.
Le chemin teacher résolu dans le JSON vise le blob Hugging Face, via le
checkpoint du runtime sous ~/.cache/onus.

Tests ajoutés au diagnostic : décompte par code 2-bit, refus tableau vide,
refus code réservé, validation shapes/dtypes, distinction cosine/amplitude.
**Cinq tests passent.** Cela teste l'outil d'enquête ; ce n'est ni un test
du futur trainer corrigé ni une certification de qualité.

## 10. Décision

Ne pas prolonger encore le dernier records-only rejeté. Garder cet artefact
comme contrôle négatif. Repartir des poids préentraînés d'origine pour
retrouver des maîtres non arrondis ; cela n'est pas entraîner depuis zéro.

Priorités : contrat temporel unique, sauvegarde exhaustive et reprise
exacte, preuve d'apprentissage des codes, puis optimisation globale
fenêtrée et écoute. Les phases, budgets, critères d'arrêt et références
sont dans le [plan V7](TERNARY_QUALITY_RECOVERY_PLAN.md).
