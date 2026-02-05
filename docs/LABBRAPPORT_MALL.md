# Sammanfattning av Kod och Metodik
## Mall för Labbrapport: Jämförelse av Analytisk och Numerisk Invers Kinematik

---

## 1. TEORI

### 1.1 Robotmodell och Denavit-Hartenberg-parametrar

Projektet modellerar en 6-frihetsgraders (6-DOF) robotarm med hjälp av **Modified Denavit-Hartenberg (MDH)**-konventionen. Varje led i robotkedjan beskrivs med fyra parametrar: `a` (länkförskjutning), `α` (länkvridning), `d` (ledförskjutning) och `θ₀` (ledvinkeloffset).

**DH-parametertabell:**

| Led | a (m) | α (rad) | d (m) | θ₀ (rad) |
|-----|-------|---------|-------|----------|
| 1   | 0     | -π/2    | 1.0   | 0        |
| 2   | 1.0   | 0       | 0     | -π/2     |
| 3   | 0     | -π/2    | 0     | π/2      |
| 4   | 0     | π/2     | 1.0   | 0        |
| 5   | 0     | -π/2    | 0     | 0        |
| 6   | 0     | 0       | 0.5   | 0        |

**Ledvinkelgränser:**

| Led | Min (rad) | Max (rad) |
|-----|-----------|-----------|
| 1   | -π        | π         |
| 2   | -π/2      | π/2       |
| 3   | -π        | π         |
| 4   | -π        | π         |
| 5   | -π/2      | π/2       |
| 6   | -π        | π         |

### 1.2 Framåtkinematik (FK)

Framåtkinematik beräknar robotarmens slutposition och orientering givet alla ledvinklar. För varje led beräknas en 4×4 homogen transformationsmatris enligt MDH-konventionen:

```
T_i = Rot_x(α_{i-1}) × Trans_x(a_{i-1}) × Rot_z(θ_i) × Trans_z(d_i)
```

Den totala transformationsmatrisen från bas till verktyg erhålls genom att multiplicera alla ledmatriser:

```
T_0→6 = T_1 × T_2 × T_3 × T_4 × T_5 × T_6
```

Resultatet är en 4×4 matris där:
- De övre vänstra 3×3 elementen representerar rotationsmatrisen R
- De övre högra 3×1 elementen representerar positionsvektorn P

### 1.3 Invers Kinematik (IK)

Invers kinematik löser det omvända problemet: givet en önskad position och orientering, beräkna vilka ledvinklar som krävs.

#### 1.3.1 Analytisk Lösning

Den analytiska metoden använder **geometrisk dekomposition** som delar upp problemet i två delar:

**Steg 1: Handledscentrum**
```
P_handled = P_mål - d₆ × R_mål[:, 2]
```
där `d₆ = 0.5 m` är avståndet från led 6 till verktyget.

**Steg 2: Armposition (led 1-3)**
- Led 1: `θ₁ = atan2(y_handled, x_handled)`
- Led 2-3: Lösning via cosinussatsen:
  ```
  r = √(x² + y²)
  s = z - d₁
  cos(θ₃) = (r² + s² - a₂² - d₄²) / (2 × a₂ × d₄)
  θ₂ = atan2(s, r) - atan2(k₂, k₁) - π/2
  ```

**Steg 3: Handledsorientation (led 4-6)**
- Beräkna rotationsmatris för handleden: `R_handled = R₀₋₃ᵀ × R_mål`
- Extrahera ZYZ Euler-vinklar:
  ```
  θ₅ = arccos(R₃₃)
  θ₄ = atan2(R₂₃, R₁₃)
  θ₆ = atan2(R₃₂, -R₃₁)
  ```
- Speciell hantering för singularitet (gimbal lock) när θ₅ ≈ 0 eller π

**Egenskaper:**
- Deterministisk (ger alltid samma resultat)
- Maskinprecision (~10⁻¹⁵ m fel)
- Snabb (~0.3 ms per lösning)
- Kan överskrida ledgränser

#### 1.3.2 Numerisk Lösning (Damped Least Squares)

Den numeriska metoden använder **Damped Least Squares (DLS)** med adaptiv dämpning:

**Algoritm:**
```
Initiering: q = q₀, λ = 0.5, steg = 0.25

För varje iteration (max 300):
  1. Beräkna FK: T_aktuell = FK(q)
  2. Beräkna 6D-fel: e = [positions_fel; rotations_fel]
  3. Om ||e|| < 10⁻⁶: KONVERGERAT ✓
  4. Beräkna Jacobian: J (6×6 matris)
  5. DLS-steg: Δθ = Jᵀ(JJᵀ + λ²I)⁻¹e
  6. Uppdatera: q_ny = q + steg × Δθ
  7. Klipp till ledgränser
  8. Adaptiv justering:
     - Om felet minskat: λ → 0.7λ, steg → 1.1×steg
     - Annars: λ → 2λ, steg → 0.5×steg
```

**Multi-restart strategi:**
Om konvergens misslyckas, prövas nya startpunkter (upp till 3 försök):
1. Användarspecificerad gissning
2. Nollställning
3. Slumpmässig konfiguration

**Egenskaper:**
- Iterativ förfining
- Noggrannhet ~10⁻⁷ m
- Långsammare (~8 ms per lösning)
- 99.96% konvergensgrad
- Respekterar ledgränser bättre

### 1.4 Manipulerbarhet (Yoshikawa-index)

Manipulerbarheten mäter hur "bra" en robotkonfiguration är för att utföra rörelser:

```
w = √(det(J × Jᵀ))
```

där J är Jacobianmatrisen. Höga värden indikerar god rörlighet i alla riktningar, medan låga värden (nära singularitet) indikerar begränsad rörlighet.

### 1.5 Statistiska Testmetoder

**Welch's t-test:**
Jämför medelvärden mellan två grupper med olika varianser:
```
t = (μ₁ - μ₂) / √(s₁²/n₁ + s₂²/n₂)
```

**Mann-Whitney U-test:**
Icke-parametriskt test som jämför rankade värden, lämpligt när data inte är normalfördelad.

**Cohen's d (effektstorlek):**
```
d = (μ₁ - μ₂) / s_poolad
```
Tolkning: |d| < 0.2 = försumbar, 0.2-0.5 = liten, 0.5-0.8 = medel, > 0.8 = stor

---

## 2. FÖRSÖKSUPPSTÄLLNING

### 2.1 Hårdvara och Mjukvara

**System:**
- Dator med modern CPU (frekvens mäts under körning)
- Python 3.x med NumPy, SciPy och psutil för systemövervakning

**Mjukvarumoduler:**
```
6-DOF_V.3/project_1/
├── kinematics/           # Kärnmoduler för kinematik
│   ├── settings.py       # Robotkonfiguration (DH-parametrar, gränser)
│   ├── analytical_solver.py  # Analytisk IK-lösare
│   ├── numerical_solver.py   # Numerisk DLS-lösare
│   ├── FK_chain.py       # Framåtkinematik
│   └── helper_func.py    # Hjälpfunktioner
├── benchmark/            # Prestandamätning
│   ├── bench_func.py     # Huvudfunktioner för benchmark
│   ├── benchmark_analytical.py
│   └── benchmark_numerical.py
├── analysis/             # Statistisk analys
│   ├── solver_comparison.py  # Jämförelserapport
│   └── plot_comparison.py    # Visualisering
└── target_gen/           # Generering av målpositioner
```

### 2.2 Testdata

**Målpositioner:**
- 100 000 slumpmässigt genererade målposer inom robotens arbetsrum
- Positioner inom sfäriskt arbetsrum (radie ~2.5 m)
- Orientering som slumpmässiga rotationsmatriser

**Arbetsrumsindelning (9 regioner):**

| Radie      | Låg höjd (z<0.5) | Medelhöjd (0.5≤z<1.5) | Hög höjd (z≥1.5) |
|------------|------------------|------------------------|-------------------|
| Inre (<1.0)| inner_low        | inner_mid              | inner_high        |
| Mellan (1.0-1.8)| middle_low  | middle_mid             | middle_high       |
| Yttre (>1.8)| outer_low       | outer_mid              | outer_high        |

### 2.3 Framgångskriterier

En lösning räknas som framgångsrik om:
- **Positionsfel** < 1 mm (0.001 m)
- **Rotationsfel** < 0.01 rad (~0.57°)
- **Ledvinkelöverensstämmelse** med definierade gränser

---

## 3. METOD

### 3.1 Benchmarkprocedur

**Övergripande flöde:**
```
1. Ladda 100 000 målpositioner från arkiv
2. För varje lösare (analytisk/numerisk):
   a. Kör benchmark i batchar om 1000 mål
   b. Mät tid, fel och systemresurser per batch
   c. Spara resultat till .npy-fil
3. Generera jämförelserapport med statistiska tester
4. Skapa visualiseringar
```

### 3.2 Mätparametrar

**För varje lösning mäts:**

| Parameter | Beskrivning | Enhet |
|-----------|-------------|-------|
| Positionsfel | Euklidiskt avstånd mellan beräknad och mål | meter |
| Rotationsfel | Vinkelmagnitud från rotationsmatrisskillnad | radianer |
| Beräkningstid | Tid för att lösa IK | sekunder |
| Ledöverensstämmelse | Om alla ledvinklar är inom gränser | bool |
| Ledmarginal | Minsta avstånd till någon ledgräns | radianer |
| Manipulerbarhet | Yoshikawa-index för konfigurationen | dimensionslös |

**Endast för numerisk lösare:**

| Parameter | Beskrivning | Enhet |
|-----------|-------------|-------|
| Iterationer | Antal DLS-iterationer | antal |
| Konvergerad | Om lösaren konvergerade | bool |
| Omstarter | Antal multi-restart försök | antal |

**Systemmätningar per batch:**

| Parameter | Beskrivning | Enhet |
|-----------|-------------|-------|
| CPU-frekvens | Aktuell processorfrekvens | MHz |
| Minnesanvändning | Förändring i RAM-användning | MB |
| CPU-utnyttjande | Procentuell processorbelastning | % |
| Genomströmning | Antal lösningar per sekund | solves/s |
| MFLOPS | Uppskattade flyttalsoperationer per sekund | miljoner |

### 3.3 Beräkning av Fel

**Positionsfel:**
```python
pos_error = ||P_beräknad - P_mål||₂
```

**Rotationsfel:**
```python
R_diff = R_beräknad × R_målᵀ
rotation_error = ||rotation_vector(R_diff)||₂
```
där `rotation_vector` extraherar rotationsvektorn (vinkel×axel) från rotationsmatrisen.

### 3.4 Statistisk Analys

**Deskriptiv statistik:**
- Medelvärde, standardavvikelse
- Median, 25:e, 75:e, 95:e och 99:e percentilen
- Interkvartilomfång (IQR)
- Min och max

**Hypotestestning:**
För varje mätparameter utförs:
1. Welch's t-test (för normalfördelad data)
2. Mann-Whitney U-test (icke-parametriskt)
3. Cohen's d effektstorlek

**Signifikansnivåer:**
- *** p < 0.001 (högt signifikant)
- ** p < 0.01 (signifikant)
- * p < 0.05 (marginellt signifikant)
- ns p ≥ 0.05 (ej signifikant)

### 3.5 Visualisering

Följande plottar genereras:
1. **Felhistogram** - Fördelning av positions- och rotationsfel (log-skala)
2. **Tidsfördelning** - Beräkningstid per lösning
3. **Arbetsrumsanalys** - Värmekarta över prestanda i olika regioner
4. **Ledmarginalanalys** - Distribution av marginal till ledgränser
5. **Manipulerbarhetsfördelning** - Jämförelse av konfigurationskvalitet

### 3.6 Dataflödesdiagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDATA                                       │
├─────────────────────────────────────────────────────────────────┤
│  Målarkiv (100k positioner)      Robotkonfiguration (DH-param.) │
│          ↓                                  ↓                    │
│  target_poses.npy                    settings.py                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK                                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐          ┌─────────────────┐               │
│  │ Analytisk IK    │          │ Numerisk IK     │               │
│  │ benchmark_      │          │ benchmark_      │               │
│  │ analytical.py   │          │ numerical.py    │               │
│  └────────┬────────┘          └────────┬────────┘               │
│           ↓                            ↓                         │
│  analytical_results.npy      numerical_results.npy               │
│  (pos_err, rot_err,          (pos_err, rot_err, time,           │
│   time, joints, etc.)         iterations, restarts, etc.)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYS                                        │
├─────────────────────────────────────────────────────────────────┤
│  solver_comparison.py                                            │
│  ├── Laddar båda resultatfiler                                   │
│  ├── Utför statistiska tester (t-test, U-test, Cohen's d)        │
│  ├── Beräknar sammanfattande statistik                           │
│  └── Genererar textrapport                                       │
│           ↓                                                      │
│  solver_comparison_report.txt                                    │
│  solver_comparison_data.npy                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  VISUALISERING                                   │
├─────────────────────────────────────────────────────────────────┤
│  plot_comparison.py                                              │
│  ├── Felhistogram (log-skala)                                    │
│  ├── Tidsfördelningar                                            │
│  ├── Arbetsrumsvärmekarta                                        │
│  └── Manipulerbarhetsplot                                        │
│           ↓                                                      │
│  data/plots/*.png                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. SAMMANFATTANDE KODSTRUKTUR

### 4.1 Modulberoenden

```
kinematics/
├── settings.py         ← Robotparametrar (ingen import)
├── helper_func.py      ← Matematiska hjälpfunktioner (numpy)
├── FK_chain.py         ← Framåtkinematik (helper_func, settings)
├── analytical_solver.py ← Analytisk IK (helper_func, settings, FK_chain)
└── numerical_solver.py  ← Numerisk IK (scipy, helper_func, settings, FK_chain)

benchmark/
├── bench_func.py       ← Huvudbenchmark (kinematics, psutil, scipy)
├── benchmark_analytical.py ← Kör analytisk (bench_func)
└── benchmark_numerical.py  ← Kör numerisk (bench_func)

analysis/
├── solver_comparison.py ← Statistik (scipy.stats, numpy)
└── plot_comparison.py   ← Plottar (matplotlib, numpy)
```

### 4.2 Nyckelklasser och Funktioner

**kinematics/analytical_solver.py:**
- `analytical_ik_solve(target_T)` → `(angles, success)`
- `analytical_ik_solve_detailed(target_T)` → `(angles, manipulability, within_limits)`

**kinematics/numerical_solver.py:**
- `numerical_ik_solve(target_T, initial_guess)` → `(angles, success)`
- `numerical_ik_solve_detailed(target_T, initial_guess)` → `dict` med alla mätvärden

**benchmark/bench_func.py:**
- `measure_batch_performance(targets, solver_type, ...)` → `dict` med resultat
- `compute_summary_statistics(...)` → `dict` med statistik
- `get_cpu_info()` → `dict` med CPU-information

**analysis/solver_comparison.py:**
- `load_results(path)` → resultatdatastruktur
- `compare_solvers(analytical, numerical)` → jämförelserapport

---

## 5. TYPISKA RESULTAT

### 5.1 Prestandajämförelse (100 000 mål)

| Mätparameter | Analytisk | Numerisk | Bäst |
|--------------|-----------|----------|------|
| Positionsfel | ~2×10⁻¹⁵ m | ~3×10⁻⁷ m | Analytisk |
| Rotationsfel | ~2×10⁻¹⁵ rad | ~3×10⁻⁷ rad | Analytisk |
| Beräkningstid | ~0.3 ms | ~8 ms | Analytisk |
| Genomströmning | ~2500/s | ~120/s | Analytisk |
| Framgångsgrad | 100% | 99.96% | Analytisk |
| Ledöverensstämmelse | ~13% | ~27% | Numerisk |
| Manipulerbarhet | ~6.0 | ~6.0 | Oavgjort |

### 5.2 Slutsatser

- **Analytisk lösare** är överlägsen för noggrannhet och hastighet
- **Numerisk lösare** är bättre på att respektera ledgränser
- Valet beror på applikationens krav (precision vs. fysisk realiserbarhet)

---

*Denna mall är genererad från kodbasen och kan användas som grund för att skriva teori-, metod- och försöksuppställningssektionerna i en labbrapport.*
