# V7 — rapport de continuation et autopsie finale

> Corrigendum : lire le [registre V9](TERNARY_V9_EVIDENCE_2026-09-25.md)
> avant de réutiliser ce diagnostic. Les symlinks ne prouvaient pas une fuite ;
> le profil 256 ne couvrait pas les trajectoires générées ; les audits V8
> portent `heldout=false` ; les champs mémoire historiques sont en GiB.
> Les échecs mesurés restent conservés, sans attribuer une cause unique.

Date : 25 septembre 2026  
Modèle : Stable Audio 3 Medium / DiT MLX  
Périmètre réellement testé : bloc 0, 7 projections, environ 56,6 M codes  
Statut : **aucun modèle final accepté**.

Ce rapport clôt les essais V7 exécutés après le rapport du 24 septembre. Il
ne transforme pas un gate candidat en gate release : toutes les métriques
ci-dessous sont des audits records rechargés, avec le corpus et le runtime
déclarés dans les manifestes correspondants.

## Décision

Le meilleur point de départ reste le candidat TTQ uniforme issu du bloc 0 :
velocity mean/min `0,96620 / 0,82497`, terminal mean environ `0,85219`.
Il passe le gate de recherche, mais pas le gate de release. Le pire prompt
reste sous la cible de travail et la stabilité terminale est insuffisante.

La cascade 0–3 puis 0–23 est donc **interdite**. Aucun artifact ≤500 MB,
aucun DiT complet 100 % ternaire et aucune réussite audio ne doivent être
annoncés à partir de V7.

## Expériences supplémentaires

| Expérience | Velocity mean/min | Terminal mean | Décision |
|---|---:|---:|---|
| TTQ uniforme, meilleur candidat | 0,96620 / 0,82497 | 0,85219 | candidat, release refusé |
| Focus prompt funk | 0,96690 / 0,81123 | 0,84590 | rejet : le pire cas baisse |
| Bloc local, hidden-state, 256 updates | 0,96071 / 0,80994 | 0,84690 | rejet |
| Dense master hybride, 64 updates | 0,91095 / 0,65412 | 0,76163 | rejet sévère |
| Dense direct, soft-to-hard, 112 updates | 0,95157 / 0,66376 | 0,80519 | rejet |
| Grille seuil statique, meilleur facteur | 0,86622 / 0,54731 | — | rejet |
| Seuil TTQ appris, LR `1e-3` | 0,90115 / 0,57934 | — | rejet |
| Seuil seul, niveaux gelés | 0,96620 / 0,82497 | — | aucun gain |
| Composition locale `ff2 + self_qkv` | 0,96372 / 0,83604 | — | min meilleur, moyenne pire |
| Polish rollout de cette composition | 0,96555 / 0,67788 | 0,85186 | rejet |

Les gates de candidate sont parfois vrais, mais le champ `release` reste faux.
Le polish rollout a consommé jusqu’à environ 11,1 GB Metal et n’a modifié
qu’environ 110 codes : sa loss oscillait et ne prédisait pas la qualité
rechargée. Le full rollout 4 étapes reste sous la garde de 12 GB ; 8 étapes
et certains groupes larges la dépassent. La mémoire n’est donc pas la cause
principale de l’échec V7.

Preuves principales :

- module swaps : `output/sample-expertise-pilot/ternary-quality-v7-20260925/module-swap-audit-candidate-vs-affine.json` ;
- swaps avec le bloc local : `.../module-swap-audit-candidate-vs-local.json` ;
- candidate TTQ : `.../ttq-block0-teacher-anchor-64updates-v1/` ;
- bloc local : `.../local-block0-ttq-hidden-smooth-256-v1/` ;
- rollout pondéré : `.../full-rollout-targets-candidate-weighted-funk-v1/`.

## Pourquoi V7 échoue encore

### 1. Le forward dur est discret, l’optimisation ne l’est pas

La loss peut diminuer en bougeant les scales ou les seuils sans sélectionner
les bons codes. La version threshold-only confirme que bouger `τ` sans
réorganiser les affectations ne gagne rien. À l’inverse, partir d’un master
dense fait beaucoup baisser la loss mais choisit une nouvelle projection
ternaire trop éloignée du comportement utile.

Le problème n’est donc plus « le seuil manque » : c’est un problème de choix
combinatoire des codes, avec un surrogate STE qui ne fournit pas le bon signal
près des frontières.

### 2. Le warm-start dequantifié et le master dense sont deux mauvais compromis

Le warm-start depuis un record conserve le comportement du candidat mais le
master FP32 déjà perdu ne peut plus récupérer l’information arrondie. Le dense
master conserve l’information, mais sa projection TTQ initiale détruit trop la
fonction. Les deux chemins ont été mesurés ; aucun ne justifie une nouvelle
longue QAT identique.

### 3. La moyenne masque un pire prompt destructeur

Le focus sur le prompt funk améliore la moyenne et dégrade le minimum. La
composition de modules fait l’inverse partiellement. Il faut sélectionner sur
un ensemble validation indépendant et sur une frontière de Pareto
`(mean, min, terminal, min_terminal)`, pas exporter la dernière loss.

### 4. Le rollout ne corrige pas la projection locale

Les cibles pondérées, le teacher rollout et le polish 4 étapes n’ont pas
récupéré le pire cas. La re-sérialisation et l’exécution réelle changent la
géométrie suffisamment pour annuler la baisse de loss. Le rollout doit devenir
un contrôle de sélection, pas le premier mécanisme d’apprentissage.

### 5. Hadamard ne résout pas une ternarisation de poids par lui-même

Les essais Hadamard étaient fonctionnellement contrôlés, mais la rotation
directe des poids partait d’une projection déjà mauvaise et dégradait souvent
le bloc. Les travaux récents HadaNorm/SpinQuant concernent surtout la gestion
des outliers d’activation ou des rotations de représentation. Ils ne prouvent
pas qu’un poids en base tournée est un poids {-1,0,+1} utile dans le kernel
réel.

## Recherche récente intégrée

- [Post-Training Quantization for Audio Diffusion Transformers](https://arxiv.org/abs/2510.00313) : les amplitudes d’activation changent avec le timestep ; le papier utilise un lissage time-aware par canal d’entrée et distingue calibration dynamique et statique. C’est directement pertinent pour les activations audio, pas une preuve de ternarisation stricte des poids.
- [HadaNorm](https://arxiv.org/abs/2506.09932) : centrage, rescaling et Hadamard réduisent des outliers d’activation ; à traiter comme une ablation d’activation, non comme un remplacement du solveur de codes.
- [QuEST](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_QuEST_Low-bit_Diffusion_Model_Quantization_via_Efficient_Selective_Finetuning_ICCV2025_paper.html) : sélection des couches sensibles et supervision locale/globale ; cela correspond mieux aux swaps contrôlés que la QAT globale aveugle.
- [TQ-DiT](https://arxiv.org/abs/2502.04056) et [TFMQ-DM](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_TFMQ-DM_Temporal_Feature_Maintenance_Quantization_for_Diffusion_Models_CVPR_2024_paper.html) : calibration dépendante du temps et maintien des features temporelles.
- [TerDiT](https://arxiv.org/abs/2405.14854) et [BitNet b1.58](https://arxiv.org/abs/2504.12285) : les résultats ternaires les plus convaincants sont obtenus avec un entraînement natif ou une architecture pensée pour le ternaire. Ils rendent nécessaire un test de faisabilité, mais ne valident pas une PTQ locale de Stable Audio 3.

## Conclusion opérationnelle

La prochaine tentative doit optimiser des décisions ternaires discrètes à
partir du master dense, avec calibration d’activations par timestep, puis
évaluer chaque état après reserialization. Les codes seuls ne suffisent pas :
les niveaux, seuils, biais, provenance, validation et budget mémoire doivent
être dans le même contrat.

Le plan exécutable est désormais [V8](TERNARY_QUALITY_RECOVERY_PLAN_V8.md).

## P0–P2 exécutés après ce rapport

### P0 : provenance réellement corrigée

Le premier préflight a révélé que le `dataset-contract.json` embarqué dans le
cache V7 déclarait le répertoire `.../authorized-independent-sftvoices-v1/train`
mais contenait des chemins résolus vers `broad-music` et `universal-dataset`.
Le cache n'était donc pas une preuve de corpus unique.

Le contrat V8 externe est maintenant construit et validé avant toute charge
MLX : digest `1d2bd975f6dad798ac105bbbf29ce42135b7daea19a3c4ee78e3776a6cc04130`,
`434` train, `33` validation, `41` test, `508` latents, zéro chevauchement
latent/parent. Les symlinks sont acceptés seulement comme chemins logiques du
split ; leur cible réelle est hashée. Le builder et l'auditeur peuvent exiger
ce contrat avec `--require-dataset-contract`.

### P1 : profil d'activation time-aware

Le profil réel du bloc 0 couvre 7 projections, 16 prompts, 8 sigmas et 2
réplicas (`256` états), avec un pic Metal de `3,68 GB`. La moyenne du p99
absolu d'entrée varie comme suit entre sigma haut et bas :

| Projection | p99 moyen bas → haut | Lecture |
|---|---:|---|
| `self_attn.to_qkv` | `2,31 → 2,38` | variation modérée |
| `self_attn.to_out` | `0,46 → 0,25` | variation inverse forte |
| `cross_attn.to_q` | `0,46 → 0,58` | variation nette |
| `cross_attn.to_kv` | `4,995 → 4,995` | quasi constant |
| `cross_attn.to_out` | `0,017 → 0,063` | variation ×3,7 |
| `ff.ff.0.proj` | `0,38 → 0,29` | variation modérée |
| `ff.ff.2` | `1,39 → 0,39` | variation ×3,6 |

Le résultat invalide le seuil unique par couche, mais ne justifie pas une
reconstruction locale indépendante du bloc.

### P2 : première sélection discrète, rejetée

Un solveur borné a testé cinq seuils par groupe, ajusté `s+`/`s-` avec les
RMS d'activation et conservé le record V7 comme option de référence. Il a
modifié des millions de codes et réduit l'erreur de poids pondérée, mais le
benchmark du forward réel s'est effondré :

| Record | Velocity mean/min train sélectionné | Validation mean/min | Terminal validation mean/min |
|---|---:|---:|---:|
| source V7 | `0,96946 / 0,89541` | `0,95833 / 0,87173` | `0,76996 / 0,64229` |
| activation-aware max | `0,86667 / 0,49690` | `0,84760 / 0,66785` | `0,44607 / 0,28554` |

Conclusion P2 : **l'importance d'activation diagonale et la MSE de poids ne
sont pas l'objectif de qualité**. Le prochain solveur doit proposer des flips
à partir du master dense, mais accepter/rejeter chaque proposition sur la
sortie du bloc et un mini-ensemble held-out, avec le sampler terminal comme
contrôle. Le record activation-aware est rejeté et ne sera jamais réutilisé.

### Scoring bloc puis full-DiT : le filtre local ne suffit toujours pas

Le cache `h_in → h_out` a permis de tester chaque swap sur 128 états sans
relancer le sampler complet. Le meilleur swap local est `ff.2` : cosine bloc
`0,98217 / 0,96097` mean/min, contre `0,95854 / 0,93147` pour la baseline.
Mais le suffixe du DiT amplifie cette modification : velocity full-DiT tombe à
`0,95741 / 0,77588`. Les autres résultats full-DiT sont :

| Composition | Velocity mean/min train sélectionné | Décision |
|---|---:|---|
| source V7 | `0,96946 / 0,89541` | référence |
| swap `ff.2` | `0,95741 / 0,77588` | rejet |
| swap `self_attn.to_out` | `0,95736 / 0,87136` | rejet, moyenne en baisse |
| swaps `ff.2 + self_attn.to_out` | `0,95565 / 0,83130` | rejet |

La règle V8 est renforcée : sortie de bloc = filtre de coût, jamais gate de
promotion. Toute décision doit repasser le full-DiT, puis la validation
held-out et le terminal avant d'accepter même un seul lot de codes.
