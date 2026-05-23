# Dynamoscope — Manuel complet

> Outil de recherche pour analyser la vidéo comme **trajectoire dans un espace d'états latents**.
> Manuel niveau débutant à licence, avec toutes les mathématiques expliquées et leurs implications.

---

## Table des matières

1. [Motivation et objectif](#1-motivation-et-objectif)
2. [Différenciation](#2-différenciation)
3. [Concept fondateur](#3-concept-fondateur)
4. [Implications philosophiques et épistémologiques](#4-implications)
5. [Architecture du projet](#5-architecture)
6. [Mathématiques expliquées](#6-mathématiques)
7. [Fonctions et modules — explication détaillée](#7-fonctions-et-modules)
8. [Le papier scientifique commenté](#8-le-papier)
9. [Résultats et leur signification](#9-résultats)
10. [Ce que ce projet apporte et ouvre](#10-implications-futures)
11. [Glossaire](#glossaire)

---

## 1. Motivation et objectif

### Pourquoi ce projet ?

Une vidéo n'est habituellement traitée comme une suite d'images indépendantes — chacune analysée pour son contenu visuel : objets, personnes, scènes. Mais une vidéo possède une **dimension temporelle** essentielle. Entre deux frames, il y a un mouvement, une évolution, une dynamique.

L'hypothèse fondatrice de Dynamoscope est simple :

> **Une vidéo peut être vue comme un point qui se déplace dans un espace abstrait au fil du temps. Ce point trace une trajectoire. La forme de cette trajectoire révèle la nature du système qui a produit la vidéo.**

Cette vision n'est pas qu'esthétique : elle s'appuie sur la théorie des systèmes dynamiques, qui décrit depuis un siècle des objets très variés (pendules, climat, économie, cellules biologiques) par leurs trajectoires dans un **espace de phases**. Le but de Dynamoscope est d'appliquer ce vocabulaire à la vidéo, et d'en faire un **microscope dynamique** pour observer les structures cachées du temps visuel.

### Objectif concret

Construire un outil qui :

1. **Encode** chaque frame d'une vidéo en un vecteur de nombres (latent).
2. **Reconstruit** la trajectoire 3D de ces vecteurs.
3. **Mesure** des invariants dynamiques : régularité, chaos, topologie, structure causale.
4. **Compare** ces mesures entre plusieurs vidéos et plusieurs modèles d'encodage.
5. **Visualise** la trajectoire en 3D pour interprétation humaine.

### Public visé

- Chercheurs en représentation latente, systèmes dynamiques, world models.
- Créateurs d'images / vidéo qui veulent un retour quantitatif sur leur production.
- Étudiants en physique, informatique, philosophie de la perception.

---

## 2. Différenciation

### Par rapport aux outils standards d'analyse vidéo

| Outil classique | Dynamoscope |
|---|---|
| Détecter objets, scènes | Mesurer la **structure dynamique** |
| Image-par-image | Vidéo comme **trajectoire entière** |
| Output : labels (chat, chien) | Output : **régime** (chaotique, périodique, fixé) |
| Métriques perceptuelles | Métriques de **systèmes dynamiques** |
| Une seule "vérité" | **Multiplicité de vues** également valides |

### Par rapport à la recherche en world models

| World models classiques (Dreamer, JEPA) | Dynamoscope |
|---|---|
| Optimisent prédictibilité | Diagnostique **richesse dynamique** |
| Un seul encoder principal | **13 modèles** comparables |
| Pertes opaques | **Diagnostic transparent** par axe |
| Pas d'analyse topologique | **Topologie** des trajectoires |

### Le double pipeline

Dynamoscope est unique en proposant **deux paradigmes côte à côte** :

- **Pipeline neural** : ResNet50, DINOv2, V-JEPA — encodeurs profonds entraînés sur des millions d'images.
- **Pipeline classique** : 12 observables physiques (luminance, mouvement, entropie...) + plongement de Takens, sans aucun apprentissage.

Et les fait dialoguer : le **probe d'observables** mesure à quel point chaque dimension du neural latent peut être expliquée par les observables classiques (R² de régression). Notre découverte : **V-JEPA est l'encodeur le plus "physique"** — 57% de ses dimensions sont linéairement explicables par les observables.

---

## 3. Concept fondateur

### L'espace de phases

Prenez un pendule simple. Pour le décrire à chaque instant, il suffit de connaître :
- Sa position angulaire θ
- Sa vitesse angulaire dθ/dt

Ces deux nombres définissent un **point** dans un plan : c'est son **espace de phases**. Au cours du temps, ce point se déplace et trace une **trajectoire** (typiquement une ellipse pour un pendule sans frottement).

Les physiciens ont compris au XIXe siècle que cette représentation est plus riche que la description temporelle directe : la **forme géométrique** de la trajectoire encode toute la dynamique du système.

### Le théorème de Takens (1981)

Floris Takens a démontré un résultat remarquable :

> **Si on ne connaît qu'une seule mesure d'un système (par exemple, la position x au cours du temps), on peut reconstruire l'attracteur entier en prenant des "copies décalées" de cette mesure.**

Formellement, pour une série temporelle scalaire s(t), on forme :

$$\vec{x}(t) = \big(s(t),\ s(t+\tau),\ s(t+2\tau),\ \ldots,\ s(t+(m-1)\tau)\big)$$

où τ est un **délai** bien choisi et m une **dimension de plongement** suffisamment grande (m ≥ 2·d + 1 où d est la dimension topologique de l'attracteur).

C'est ce que Dynamoscope fait dans le pipeline classique : à partir d'une seule observable scalaire (par exemple, le mouvement par frame), il reconstruit une trajectoire dans un espace m-dimensionnel, **diffeomorphe à l'attracteur réel du système**.

### Pourquoi ça marche pour la vidéo ?

Une vidéo est produite par **un système** : une caméra qui filme, ou un algorithme qui génère. Ce système a sa propre dynamique. Les frames sont des observations de ce système. Selon Takens, on peut reconstruire la dynamique sans connaître le système — juste en regardant la séquence d'observations.

Dynamoscope va plus loin : au lieu d'une seule observable scalaire, il utilise **12 observables physiques** + des **encodeurs profonds** (qui produisent des centaines à milliers de scalaires par frame) pour reconstruire un espace de phases beaucoup plus riche.

---

## 4. Implications philosophiques et épistémologiques

### Pluralisme de représentation

Dynamoscope incarne une thèse explicite :

> **Aucune représentation unique ne capture toute la vérité d'un signal dynamique.**

Chaque encodeur, chaque réducteur de dimension, chaque métrique extrait un aspect différent. Le rôle de l'outil n'est pas de fournir LA bonne vue mais d'**exposer la multiplicité** des vues valides.

C'est un héritage direct de la **mécanique quantique** (l'observateur fait partie du système observé) et de la **philosophie des sciences post-positiviste** (Kuhn, Feyerabend : plusieurs paradigmes peuvent décrire le même phénomène).

### Le pluralisme classique-vs-neural

Le double pipeline (classique + neural) matérialise cette thèse :

- **Le classique** est interprétable, lent à l'invention, fidèle à la physique des observables.
- **Le neural** est opaque, expressif, capture des invariances sémantiques.

Aucun n'est supérieur en absolu. L'expérience de "distillation" (§7.7 du papier) montre que **les forcer à converger échoue systématiquement** : leurs géométries latentes sont mutuellement exclusives. Cette incompatibilité est elle-même un résultat scientifique : elle indique deux modes irréductibles de représentation.

### Information, observateur, dynamique

Dynamoscope adhère implicitement à la vision de **John Archibald Wheeler** :

> "It from bit" — toute matière est faite d'information.

Une vidéo n'est pas une "réalité" objective : c'est de l'information sur un système, mesurée par un capteur (caméra, modèle génératif), transformée par un encodeur. La trajectoire latente est une **réinscription** de cette information dans un espace où elle devient analytique. La question "que voit le modèle ?" devient "quelle est la géométrie de l'information qu'il extrait ?".

### Émergence

L'un des fils conducteurs de toute la recherche est l'**émergence** : comment des règles simples (par exemple, une équation différentielle de Lorenz) produisent des structures complexes (l'attracteur étrange). Dynamoscope est, en un sens, un microscope pour observer cette émergence dans des artefacts vidéo.

---

## 5. Architecture du projet

### Vue d'ensemble

```
┌────────────────────────────────────────────────────────────────┐
│                       FRONTEND (browser)                       │
│   Three.js 3D trajectory viz + tabs (dynamics, topology, ...)  │
└────────────────────────┬───────────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                         │
├────────────────────────────────────────────────────────────────┤
│  Ingestion (video → frames)                                    │
│  Encoders (frames → latents)         [classical OR neural]     │
│  Reduction (latents → 3D coords)      [UMAP, PCA, Isomap]      │
│  Analysis modules                                              │
│   ├ dynamics.py     (Lyapunov, RQA, correlation dim)           │
│   ├ topology.py     (persistent homology)                      │
│   ├ causal.py       (Granger, TE, PCMCI+, KSG)                 │
│   ├ spectral.py     (DMD, Koopman, power spectrum)             │
│   ├ sindy.py        (sparse equation discovery)                │
│   ├ wavelets.py     (multi-scale decomposition)                │
│   ├ regime_classifier.py (verdict 6 régimes)                   │
│   └ dna.py          (8-axis composite score)                   │
│  Training (mini RSSM + differentiable losses)                  │
│  Cross-paradigm probe (R² neural ← observables)                │
└────────────────────────────────────────────────────────────────┘
```

### Dossiers

```
backend/                Code Python serveur
  models/               TrajectoryProducer wrappers (13 modèles)
  losses/               5 pertes différentiables (Phase C + distill)
  training/             Mini RSSM + PCGrad
frontend/               UI HTML/JS/Three.js
experiments/            Scripts expérience reproductibles
results/                JSON outputs des experiments
tests/                  101 tests pytest
paper/                  Article LaTeX (~1500 lignes)
videos/synthetic/       6 vidéos procédurales benchmark
```

---

## 6. Mathématiques expliquées

### 6.1 Trajectoire dans un espace latent

**Idée vulgaire** : si chaque frame de vidéo donne un vecteur de N nombres, la suite de tous ces vecteurs trace une "ligne" dans un espace à N dimensions.

**Formellement** : pour une vidéo de T frames et un encodeur f : Frame → ℝᴺ, la trajectoire latente est :

$$Z = \big(z_1, z_2, \ldots, z_T\big),\quad z_t = f(\text{frame}_t),\quad z_t \in \mathbb{R}^N.$$

### 6.2 Réduction de dimension : PCA

**Idée vulgaire** : si chaque point a 2048 dimensions mais ne se balade que dans 3 directions principales, on peut "écraser" l'espace pour ne garder que ces 3 directions.

**Formellement** : Analyse en Composantes Principales (PCA) trouve les directions de variance maximale :

1. Centrer : Z̃ = Z − mean(Z)
2. Calculer la matrice de covariance : C = Z̃ᵀZ̃ / T
3. Diagonaliser : C = V Λ Vᵀ
4. Projeter : Z₃ = Z̃ V[:, :3] (les 3 premières colonnes de V)

Les 3 nouvelles coordonnées sont les **3 directions d'étalement maximum**.

### 6.3 UMAP

**Idée vulgaire** : UMAP préserve les **voisinages locaux**. Deux frames proches dans l'espace original le restent dans la projection 3D, même si globalement la géométrie est déformée.

UMAP minimise une **divergence** entre la structure locale haute-D et basse-D, basée sur la théorie des **simplicial sets** (McInnes, Healy, Melville 2018).

### 6.4 Théorème de Takens : plongement par délais

Voir §3 plus haut.

**Choix automatique de τ** (auto-tau) : on prend la première **valeur où l'autocorrélation passe par un minimum local** :

$$\tau^* = \arg\min_{\tau \geq 1} \big|\,\mathrm{autocorr}(s, \tau)\,\big|.$$

**Choix de m** : par défaut 3 (suffisant pour visualiser et pour la plupart des systèmes à dimension fractale < 1).

### 6.5 Exposant de Lyapunov

**Idée vulgaire** : si deux trajectoires initialement proches s'éloignent de plus en plus, le système est **chaotique**. L'exposant de Lyapunov mesure à quelle vitesse elles s'éloignent.

**Formellement** (méthode de Rosenstein, Collins & De Luca 1993) :

Pour chaque point i de la trajectoire, on trouve son plus proche voisin j (en excluant les voisins temporels par une fenêtre de Theiler W). Puis on suit comment la distance évolue :

$$d_{i,j}(t) = \|z_{i+t} - z_{j+t}\|.$$

L'exposant de Lyapunov dominant est la **pente moyenne** de ln(d_{i,j}(t)) vs t :

$$\lambda \approx \frac{d\langle \ln d_{i,j}(t)\rangle}{dt}.$$

- **λ > 0** : chaos (divergence exponentielle).
- **λ ≈ 0** : régime régulier (cycle, tore).
- **λ < 0** : convergence vers point fixe.

### 6.6 Analyse de récurrence (RQA)

**Idée vulgaire** : on regarde quand une trajectoire **repasse près d'un point déjà visité**. Si ça arrive souvent et de manière organisée (diagonales dans une matrice), le système est régulier. Sinon, chaotique ou aléatoire.

**Matrice de récurrence** :

$$R_{ij} = \mathbb{1}\big[\|z_i - z_j\| < \varepsilon\big].$$

ε est calibré pour que ~10% des paires soient récurrentes.

**Métriques RQA** :

- **Recurrence Rate (RR)** : densité globale = Σ Rᵢⱼ / N²
- **Determinism (DET)** : fraction des points sur des **diagonales de longueur ≥ 2** = prédictibilité
- **Laminarity (LAM)** : fraction sur **verticales** = phases quasi-stationnaires
- **L_max/N** : longueur de la plus longue diagonale = régularité

### 6.7 Dimension de corrélation

**Idée vulgaire** : si la trajectoire vit "à plat" (en 1D), doubler la taille de la boule autour d'un point double le nombre de points dedans. En 2D, ça quadruple. En 3D, ça octuple. La dimension de corrélation **interpole** cette idée pour des objets fractals (non-entiers).

**Formellement** (Grassberger-Procaccia 1983) :

$$C(\varepsilon) = \frac{1}{N(N-1)} \sum_{i \neq j} \mathbb{1}\big[\|z_i - z_j\| < \varepsilon\big].$$

La dimension de corrélation est :

$$d = \lim_{\varepsilon \to 0} \frac{\log C(\varepsilon)}{\log \varepsilon}.$$

En pratique, on **fit la pente** sur une plage d'ε où C(ε) suit une loi de puissance.

Pour l'attracteur de Lorenz : d ≈ 2.06 (non-entier = fractal).

### 6.8 Homologie persistante

**Idée vulgaire** : on grossit progressivement chaque point en une boule, et on regarde quand des **anneaux** (cycles) apparaissent et disparaissent. Un anneau qui "vit longtemps" est une caractéristique topologique réelle ; un anneau qui apparaît et disparaît vite est du bruit.

**Formellement** : on construit le **complexe de Vietoris-Rips** R_ε(Z) pour une suite de scales ε croissants. Chaque scale produit des **classes d'homologie** :

- H₀ : nombre de **composantes connexes**.
- H₁ : nombre de **cycles** (anneaux, trous).
- H₂ : nombre de **cavités** (sphères creuses).

Chaque classe a une **naissance** ε_b et une **mort** ε_d. La **persistance** ε_d − ε_b est sa "longueur de vie". Le **diagramme de persistance** trace toutes les paires (ε_b, ε_d).

**Entropie de persistance** : si on a k cycles avec persistances p_1, ..., p_k normalisées en p̃ᵢ = pᵢ / Σpⱼ, alors :

$$H = -\sum_i \tilde p_i \log \tilde p_i.$$

Haute entropie = plusieurs cycles d'importance comparable = topologie riche.

### 6.9 Causalité : Transfer Entropy

**Idée vulgaire** : la causalité statistique mesure si connaître le passé d'une variable X **réduit l'incertitude** sur le futur d'une variable Y (au-delà de ce que le passé de Y donne déjà).

**Formellement** (Schreiber 2000) :

$$\mathrm{TE}_{X \to Y} = \sum p(y_{t+1}, y_t^k, x_t^l) \log \frac{p(y_{t+1} \mid y_t^k, x_t^l)}{p(y_{t+1} \mid y_t^k)}.$$

Si TE > 0, X "cause" Y (au sens informationnel).

**Implementation KSG (Kraskov-Stögbauer-Grassberger)** : estimateur continu basé sur les k plus proches voisins. Plus robuste que les histogrammes pour des données continues.

### 6.10 PCMCI+

**Causal discovery sur séries temporelles multi-variées.** Combine :

- **PC algorithm** : élimine les liens via tests d'indépendance conditionnelle.
- **MCI** (Momentary Conditional Independence) : adapte la condition pour séries temporelles.

Output : graphe orienté indiquant pour chaque paire (Xᵢ → Xⱼ, lag k) si la dépendance est causale.

### 6.11 SINDy (Sparse Identification of Nonlinear Dynamics)

**Idée vulgaire** : on suppose que la dynamique latente obéit à une équation différentielle dz/dt = f(z), et on cherche f comme **combinaison sparse** de termes polynomiaux.

**Formellement** :

1. Calculer dérivée : dz/dt par différence finie.
2. Construire library Θ(z) = [1, z, z², z·z', z³, ...].
3. Résoudre : dz/dt = Ξ Θ(z) sous contrainte de sparsité (STLSQ ou Lasso).

Si SINDy retrouve l'équation de Lorenz à partir de la trajectoire latente, c'est que l'encodeur a appris une **représentation suffisamment fidèle**.

### 6.12 DMD (Dynamic Mode Decomposition) et opérateur de Koopman

**Idée vulgaire** : on essaie de **linéariser** la dynamique dans un espace plus grand (Koopman). Les modes propres de cet opérateur donnent les **fréquences** et **taux d'amortissement** du système.

**DMD algorithm** :

Soient X = [z_1, z_2, ..., z_{T-1}] et X' = [z_2, z_3, ..., z_T]. On cherche A tel que X' ≈ A X. Solution par SVD :

1. SVD : X = U Σ Vᵀ.
2. Ã = Uᵀ X' V Σ⁻¹.
3. Eigendecomposition : Ã W = W Λ.
4. Modes DMD : Φ = X' V Σ⁻¹ W.

Les eigenvalues λᵢ ∈ ℂ donnent :
- |λᵢ| = 1 : oscillation pure.
- |λᵢ| < 1 : décroissance.
- |λᵢ| > 1 : croissance.

### 6.13 Wavelets et analyse multi-échelle

**Idée vulgaire** : Fourier décompose un signal en sinusoïdes infinies. Les wavelets décomposent en **ondelettes localisées dans le temps** — chaque coefficient dit "à tel moment, à telle fréquence".

**DWT (Discrete Wavelet Transform)** : décomposition en N niveaux, chaque niveau capturant une fréquence différente.

**Scattering de Mallat** : hiérarchie de wavelets + non-linéarité + agrégation. Invariant à petites translations, stable au déformations.

### 6.14 SFA (Slow Feature Analysis)

**Idée vulgaire** : extraire les features qui varient le plus **lentement** dans le temps. Ces features ont tendance à capturer du contenu sémantique (identité de scène, objet) plutôt que du bruit haute-fréquence.

**Formellement** (Wiskott-Sejnowski 2002) :

minimiser ⟨(Δy)²⟩ sous contraintes ⟨y⟩ = 0, ⟨y²⟩ = 1.

Solution : eigendecomposition généralisée des matrices de covariance Σ et Σ_Δ.

### 6.15 Pertes différentiables (Phase C)

Nos quatre pertes :

**LyapunovMatchingLoss** : drive l'estimé Lyapunov de la trajectoire vers une cible.

$$\mathcal{L}_{\text{Lyap}} = (\hat\lambda - \lambda^*)^2.$$

**TopologyPreservationLoss** : préserve la géométrie relative (corrélation Pearson sur distances pairwise).

$$\mathcal{L}_{\text{Topo}} = 1 - \rho\big(\mathrm{vec}(D_{\text{lat}}), \mathrm{vec}(D_{\text{ref}})\big).$$

**SFASlownessRegularizer** (mode standardisé) : pénalité sur Δz normalisé.

$$\mathcal{L}_{\text{SFA}} = \frac{1}{T-1}\sum_t \|\tilde z_{t+1} - \tilde z_t\|^2,\quad \tilde z = (z - \mu)/\sigma.$$

**CausalSparsityLoss** : fit AR + L1 + DAG penalty (NOTEARS).

$$\mathcal{L}_{\text{Caus}} = \|z_{t+1} - A z_t\|^2 + \alpha \|A\|_1 + \beta\,h(A),\quad h(A) = \mathrm{tr}(e^{A \odot A}) - D.$$

**ClassicalAlignmentLoss** : aligne la géométrie neurale sur une trajectoire classique cible.

$$\mathcal{L}_{\text{align}} = 1 - \rho\big(D_{\text{neural}}, D_{\text{classical}}\big).$$

### 6.16 PCGrad (gradient surgery)

**Idée vulgaire** : si deux objectifs sont en conflit (leurs gradients pointent dans des directions opposées), on **projette** le gradient de l'un sur le **complémentaire orthogonal** de l'autre. Permet d'éviter qu'ils s'annulent.

**Algorithme** (Yu et al. 2020) :
1. Compute gᵢ pour chaque tâche.
2. Pour chaque paire (gᵢ, gⱼ) : si ⟨gᵢ, gⱼ⟩ < 0, gᵢ ← gᵢ − (⟨gᵢ, gⱼ⟩ / ‖gⱼ‖²) gⱼ.
3. Sommer et faire le pas.

### 6.17 DNA composite score

8 axes (chaos, topologie, complexité, spectral, structure, causalité, prédictibilité, regime_confidence), chacun normalisé dans [0, 1] :

$$\mathrm{DNA} = 100 \times \sum_k w_k \cdot \mathrm{axis}_k,\quad \sum_k w_k = 1.$$

Score haut = dynamique riche.

---

## 7. Fonctions et modules — explication détaillée

### `backend/ingestion.py`

`sample_frames(video_path, max_frames=200)` : extrait N frames espacées uniformément.

`save_thumbnails(frames, dir, width=240)` : sauvegarde miniatures JPG pour player.

### `backend/observables.py` (pipeline classique)

`compute_observables(frames)` : pour chaque frame, calcule 12 scalaires :
- brightness (luminance moyenne)
- contrast (écart-type luminance)
- meanR, meanG, meanB (moyennes RGB)
- motion (différence absolue inter-frame)
- flowX, flowY (centroïde du mouvement = proxy optical flow)
- entropy (Shannon sur histogramme 32 bins)
- edges (gradient Sobel moyen)
- centroidX, centroidY (centroïde pondéré luminance)

Tout est calculé en numpy + OpenCV, sans GPU.

### `backend/embedding.py`

`auto_tau(series)` : trouve premier minimum local de |autocorrelation|.

`delay_embed(series, m, tau)` : Takens, retourne (T − (m-1)τ, m).

`pca_embed(channels, keys, m)` : PCA via power iteration + déflation. m components.

`direct_embed(channels, keys)` : sélectionne K observables comme axes directs.

### `backend/encoder.py`

Encodeurs neuraux pré-entraînés frozen :
- ResNet50 (25M params, ImageNet supervisé)
- ViT-B/16 (86M, ImageNet)
- DINOv2-S/14 (22M, self-supervisé)
- CLIP ViT-B/32 (151M, contrastive image-text)
- VideoMAE-base (86M, self-supervisé sur clips 16-frame)

Chaque encodeur a `encode(frames) → (N, latent_dim) numpy array`.

### `backend/reducer.py`

`reduce_3d(latents, method='umap')` : réduit à 3D pour visualisation. Méthodes : umap, pca, isomap.

### `backend/dynamics.py`

`mutual_info_lag(series)` : choix optimal de τ via min MI.

`takens_embed(series, m, tau)` : plongement par délais (version dynamics module).

`recurrence_matrix(traj, eps)` : matrice de récurrence binaire.

`rqa_stats(R)` : RR, DET, LAM.

`lyapunov_rosenstein(traj)` : exposant de Lyapunov + courbe de divergence.

`correlation_dimension(traj)` : Grassberger-Procaccia.

`max_diagonal_ratio(R)` : Lmax / N.

`convergence_rate(coords)` : décroissance relative du rayon = signature point fixe.

`analyse_trajectory(coords)` : pipeline complet, retourne dict avec tous métriques.

### `backend/topology.py`

`persistent_homology(coords, max_dim=1)` : appelle ripser, retourne diagrammes H₀, H₁ + entropie.

### `backend/causal.py` + `causal_advanced.py`

`granger_test(x, y)` : test Granger classique (statsmodels).

`transfer_entropy(x, y)` : TE histogramme.

`transfer_entropy_ksg(x, y, k=4)` : TE KSG (Kraskov), k-NN continu.

`pcmci_discovery(data)` : PCMCI+ via tigramite.

### `backend/spectral.py`

`dmd(latents)` : Dynamic Mode Decomposition, retourne modes + eigenvalues.

`power_spectrum(series)` : FFT + |X|².

### `backend/multiscale.py`

`spectral_slope(latents)` : exposant de loi de puissance du spectre = "1/f^α".

### `backend/wavelets.py`

`dwt_decomposition(series)` : decomposition wavelet discrete par niveau.

### `backend/scattering.py`

`scattering_1d(series, J, Q)` : hiérarchie Mallat S₀/S₁/S₂.

### `backend/sindy.py`

`fit_sindy(coords, dt, order=2)` : sparse regression, library polynomiale + STLSQ.

### `backend/emergence.py`

`compute_emergence_index(coords)` : indice φ_id (Hoel) via macroscale graining + EI.

### `backend/sfa.py`

`slow_features(latents, k=3)` : k slowest features via eigendecomposition généralisée.

### `backend/regime_classifier.py`

`classify_regime(...)` : 6 régimes (fixed, noise, strange, cycle, torus, unknown) via cascade de règles sur métriques.

### `backend/dna.py`

`compute_dna(latents, coords_3d)` : 8-axis composite score + label + verdict regime.

### `backend/models/` (TrajectoryProducer protocol)

`base.py` : ABC `TrajectoryProducer` avec `produce_trajectory(frames) → (N, D)`.

`encoders.py` : wrap les 5 encodeurs.

`world_models.py` : Dreamer stub, V-JEPA2 real, RSSM stub.

`observable_producers.py` : DelayEmbedProducer, PCAObservableProducer, DirectObservableProducer.

`registry.py` : 13 modèles plug-and-play.

### `backend/losses/`

Voir §6.15. 5 modules nn.Module.

### `backend/training/`

`mini_rssm.py` : CNN encoder + GRU dynamics + transposed CNN decoder, 2.76M params.

`trainer.py` : TrainConfig + RSSMTrainer, combine recon + dynamics + Phase C + alignment losses.

`pcgrad.py` : Yu et al. 2020 surgery.

### `backend/cross_paradigm.py`

`observable_probe(z, observables)` : ridge regression, R² + top observables per dim.

`observable_probe_transfer(z_train, obs_train, z_test, obs_test)` : généralisation test.

### `backend/benchmark.py` + `benchmark_models.py`

`cross_model_benchmark(videos, models)` : matrice (video × model) avec DNA, dynamics, timings.

`render_leaderboard(result)` : ASCII table classement.

### `backend/main.py`

FastAPI routes :
- `/api/process` : upload + encode + reduce + return coords + frame URLs.
- `/api/dynamics` : Lyapunov/RQA/corr_dim.
- `/api/topology` : persistence.
- `/api/causal*` : Granger/TE/PCMCI.
- `/api/spectral` : DMD + power spectrum.
- `/api/sindy` : équations sparse.
- `/api/observables/{key}` : 12 channels.
- `/api/embed/{key}` : Takens / PCA / direct.
- `/api/regime` : verdict classifier.
- `/api/encoders` : liste modèles.

---

## 8. Le papier scientifique commenté

Le papier (`paper/dynamoscope.tex`, ~1500 lignes, 35 références) est structuré ainsi :

### §1 Introduction
Pose la vision : vidéo = trajectoire latente. Annonce contributions.

### §2 Related Work
Couvre Takens, Rosenstein, persistance, Koopman, JEPA, Dreamer, SFA, RQA.

### §3 Methods
Pipeline complet, depuis ingestion jusqu'aux métriques.

### §4 Multi-encoder integration
13 modèles registrés. Protocole TrajectoryProducer.

### §5 Diagnostic axes
8 axes du DNA + chaque métrique en détail.

### §6 Differentiable Diagnostic Losses
4 pertes Phase C + ClassicalAlignmentLoss + équations.

### §7 Empirical Validation
- §7.1-7.4 Phase D' rigorous (loss convergence, training ablation, baselines)
- §7.5 Training Length cautionary
- §7.6 Classical vs Neural (classique gagne sur DNA composite)
- §7.7 Distillation — 4 falsifications (sum, PCGrad, channel split, capacity)
- §7.8 Honest Limitations

### §8 Application : Bifurcation Detection
- §8.1 Sliding window — 4 falsifications + 1 confirmation (observable choice gives 230× spread, flowY near-perfect at +5f lag)

### §9 Conclusion

**Style du papier** : transparent sur les négatifs. Chaque expérience qui échoue est documentée avec autant de rigueur que celle qui réussit. C'est l'opposé du **publication bias** académique.

---

## 9. Résultats et leur signification

### 9.1 Classical observables battent neural encoders sur DNA

Sur 6 vidéos procédurales × 7 modèles :

| Modèle | DNA mean | Type |
|---|---|---|
| delay_entropy | 48.98 | classical |
| pca_obs | 46.07 | classical |
| delay_motion | 45.57 | classical |
| delay_brightness | 44.44 | classical |
| ResNet50 (25M) | 41.84 | neural |
| DINOv2-S (22M) | 41.58 | neural |

**Signification** : pour mesurer la **richesse dynamique** d'une vidéo, des observables physiques simples (sans GPU) battent des encodeurs profonds 25M de paramètres. Pour la classification sémantique, c'est l'inverse — mais pour la dynamique, la simplicité est mieux. C'est un résultat **contre-intuitif** qui contredit le bias actuel "neural = always better".

### 9.2 Phase C losses améliorent significativement la richesse dynamique

5 seeds × 4 régimes (smooth/periodic/chaotic) :

- Périodic DNA : +9.3 pts (p ≈ 0)
- Chaotic DNA : +4.1 pts (p = 0.008)
- Smooth DNA : −11.7 pts (over-régularisation)

**Signification** : les pertes différentielles fonctionnent **selon le régime**. Aucun loss n'est universel. Confirmation expérimentale du pluralisme représentationnel.

### 9.3 Distillation classical → neural : hybride réussi

Mini-RSSM entraîné avec `ClassicalAlignmentLoss(z, takens_motion)` :
- Gagne sur chaos (+0.35), topologie (+0.14)
- Garde predictability neurale (+0.18)
- Trade-off : spectral (−0.24)

**Signification** : on peut transférer la richesse dynamique du classique au neural via une perte de geometry-matching. Premier pas vers des **encodeurs neuro-classiques**.

### 9.4 Combiner Phase C + Distill ne marche pas

4 fixes testés, tous échouent :
- Naive sum
- PCGrad
- Channel separation
- Capacité doublée

**Signification** : l'incompatibilité est **représentationnelle**, pas optimisation/capacité. Les deux objectifs demandent des géométries latentes mutuellement exclusives. C'est un **résultat négatif fort** qui informera le design des futures architectures hybrides.

### 9.5 Bifurcation detection : flowY trouve Hopf à 5 frames du true value

12 observables testées comme inputs au delay embedding, lag varie de +5f (flowY) à +235f (meanR). **230× spread**.

**Signification** : pour détecter une transition dynamique dans une vidéo, **le choix de l'observable est dominant**. La résolution spatiale (testée à 64/128/256), la fenêtre temporelle (15-120), la métrique (4 testées), et la longueur de trail (1-24) sont **tous invariants**. Conséquence pratique : pour une nouvelle application, **investir dans le bon observable** plutôt que dans plus de paramètres.

### 9.6 Observable probe : V-JEPA le plus "physique"

| Encoder | mean R² | dims interprétables |
|---|---|---|
| V-JEPA 2 | 0.568 | 57% |
| DINOv2 | 0.543 | 51% |
| delay_motion | 0.589 | 50% |
| ResNet50 | 0.302 | 16% |

**Signification** : V-JEPA (world model) capture plus de signal physique que les encodeurs perceptuels. Confirme l'hypothèse que **l'entraînement world-model produit des représentations plus alignées avec la physique**. ResNet50 (perceptual, IL-supervisé) est le moins interprétable.

Et : **probe ne transfère pas entre régimes** (R² négatifs au test). La mapping neural→classical est **régime-spécifique**, ce qui justifie l'approche regime-aware proposée dans le framework futur.

---

## 10. Ce que ce projet apporte et ouvre

### Contributions actuelles

1. **Outil open-source** pour analyser vidéos via systèmes dynamiques, avec 13 modèles plug-and-play.
2. **Double pipeline** classique + neural, comparable directement.
3. **Différentiable losses** intégrant 4 invariants dynamiques.
4. **ClassicalAlignmentLoss** : nouvelle perte distillation classique→neural.
5. **Rigorous empirical methodology** : multi-seed, bootstrap CI, Welch t-tests, falsifications honnêtes.
6. **Datasets procéduraux** reproductibles (6 systèmes : Lorenz, VdP, Hénon, etc).
7. **Paper LaTeX** avec 8 sections + 6 tables expérimentales en §8.1 seule.
8. **101 tests pytest** couvrant tout le backend.

### Implications théoriques

- **Pluralisme paradigmatique validé** : classical et neural sont mutuellement exclusifs au gradient, mais complémentaires en information. Il faut maintenir les deux.
- **Observable choice dominates** : pour la détection dynamique, choisir le bon observable physique a plus d'impact que d'augmenter modèle/résolution/fenêtre.
- **World-model > perceptual encoder** pour alignment avec physique : V-JEPA bat ResNet50 / DINOv2 sur R² observable probe.

### Implications philosophiques

- Une vidéo n'a **pas de représentation canonique**. Chaque pipeline en extrait une géométrie, et la vraie information se trouve dans la **multiplicité comparée**, pas dans le choix d'une.
- L'**émergence** est observable empiriquement : on peut tracer le moment où une bifurcation devient visible dans une trajectoire latente.
- Le **rôle de l'observateur** est crucial — pas juste philosophiquement mais quantitativement : 230× écart entre observables différents.

### Directions ouvertes

- **Regime-aware world models** : architectures où l'état latent contient explicitement des descripteurs dynamiques (Lyapunov, topology, regime).
- **Multi-geometry latents** : différentes parties de l'état latent vivent sur différentes variétés selon le régime.
- **Real video evaluation** : Penn Action, Kinetics — tester ecological validity.
- **Tasks downstream** : classification d'action, retrieval, anomaly detection — voir si la richesse dynamique aide.
- **Hierarchical objectives** : redesigner Phase C et Distill pour qu'elles soient compatibles via une décomposition de l'objectif.

---

## Glossaire

### A
**Attractor** : ensemble de points vers lequel une trajectoire converge à long terme. Peut être un point fixe, un cycle limite, un tore, ou un attracteur étrange (fractal).

**Auto-tau** : choix automatique du délai τ pour le plongement de Takens, basé sur le premier minimum de |autocorrelation|.

### B
**Bifurcation** : changement qualitatif du comportement d'un système quand un paramètre franchit une valeur critique. Exemple : Hopf bifurcation transforme un point fixe stable en cycle limite.

**Bootstrap CI** : intervalle de confiance estimé par rééchantillonnage avec remise. 95% CI = entre 2.5e et 97.5e percentile.

### C
**Causality (Granger)** : X cause Y si connaître le passé de X aide à prédire Y au-delà de ce que le passé de Y donne déjà.

**Composite score DNA** : agrégation pondérée de 8 axes diagnostiques en un seul nombre.

**Correlation dimension** : dimension fractale estimée par Grassberger-Procaccia.

### D
**Delay embedding (Takens)** : reconstruction d'attracteur via [s(t), s(t+τ), ..., s(t+(m-1)τ)].

**Determinism (RQA)** : fraction des points de la matrice de récurrence sur des diagonales ≥ 2 = prédictibilité.

**Distillation** : transférer connaissance d'un modèle "teacher" à un "student" via une perte de matching.

**DMD** : Dynamic Mode Decomposition. Linéarise approximativement la dynamique : z_{t+1} ≈ A z_t. Spectre de A = modes + fréquences.

### E
**Embedding dim (m)** : dimension du plongement de Takens. Doit être ≥ 2·d_box + 1 (Whitney).

**Entropy (Shannon)** : −Σp log p. Mesure d'incertitude.

**Epsilon (ε)** : seuil de récurrence dans RQA. Calibré pour obtenir ~10% de paires récurrentes.

### F
**FastAPI** : framework Python pour API REST (utilisé par backend).

**flowX, flowY** : coordonnées du centroïde du mouvement inter-frame. Proxy d'optical flow.

### G
**Granger causality** : voir Causality.

**Grassberger-Procaccia** : algorithme pour estimer la dimension de corrélation.

### H
**H₀, H₁, H₂** : classes d'homologie. H₀ = composantes connexes, H₁ = cycles, H₂ = cavités.

**Hopf bifurcation** : transition point fixe ↔ cycle limite (cf Van der Pol à μ=0).

### J
**JEPA / V-JEPA** : Joint Embedding Predictive Architecture (LeCun, Bardes). Apprend à prédire dans l'espace latent sans reconstruire.

### K
**Koopman operator** : opérateur linéaire infini-dimensionnel qui agit sur les fonctions d'état. Linéarise la dynamique non-linéaire.

**KSG (Kraskov-Stögbauer-Grassberger)** : estimateur de transfer entropy basé sur k plus proches voisins.

### L
**Lyapunov exponent (λ)** : taux de divergence de trajectoires initialement proches. λ > 0 = chaos.

**Lmax/N** : longueur de la plus longue diagonale de récurrence sur N = régularité.

### M
**Mini-RSSM** : modèle RSSM compact (2.76M params) entraînable sur CPU/MPS.

**Mutual information (MI)** : mesure de dépendance statistique entre deux variables.

### N
**NOTEARS** : contrainte différentiable pour forcer une matrice à représenter un DAG.

### P
**PCA** : Principal Component Analysis. Trouve directions de variance maximale.

**PCGrad** : Gradient Surgery for Multi-Task Learning (Yu 2020). Projette orthogonal en cas de conflit.

**PCMCI+** : algorithme de découverte causale pour séries temporelles multivariées.

**Persistent homology** : suit naissance/mort de classes d'homologie sur une filtration.

**Phase space** : espace des coordonnées + vitesses. Plus généralement, espace d'état complet d'un système.

### R
**Recurrence matrix (R)** : Rᵢⱼ = 1 si ‖zᵢ − zⱼ‖ < ε. Visualise revisitations.

**Regime classifier** : assigne un label (fixed/noise/strange/cycle/torus) selon métriques.

**Ridge regression** : régression linéaire avec pénalité L2 sur coefficients.

**Rosenstein** : algorithme d'estimation de Lyapunov (1993).

**RQA** : Recurrence Quantification Analysis. Métriques sur R.

**RSSM** : Recurrent State-Space Model. Architecture world model (Hafner).

### S
**SFA** : Slow Feature Analysis (Wiskott 2002). Extrait features lentes.

**SINDy** : Sparse Identification of Nonlinear Dynamics (Brunton 2016).

**Slowness loss** : pénalité sur la vitesse de variation des latents.

**Spectral slope** : exposant de la loi de puissance du spectre de Fourier.

**STLSQ** : Sequentially Thresholded Least Squares. Algorithme sparse pour SINDy.

### T
**Takens (théorème de)** : permet reconstruction d'attracteur depuis une observation scalaire.

**Theiler window (W)** : exclut paires (i, j) avec |i-j| < W lors du calcul de Lyapunov ou récurrence, pour éviter biais de voisins temporels.

**Transfer entropy (TE)** : mesure causalité informationnelle de X vers Y.

### U
**UMAP** : Uniform Manifold Approximation and Projection. Réduction non-linéaire préservant voisinages locaux.

### V
**Van der Pol oscillator** : dx/dt = y, dy/dt = μ(1-x²)y - x. Bifurcation Hopf à μ=0.

**Vineyards** : suivi temporel des cycles de persistance via Hungarian matching.

### W
**Welch's t-test** : test de comparaison de moyennes pour variances inégales.

**Wavelet** : ondelette localisée en temps et fréquence.

### Z
**Zscore** : normalisation (x − μ) / σ.

---

*Manuel généré pour Dynamoscope, version mai 2026. Code source : https://github.com/zacccarie/dynamoscope*
