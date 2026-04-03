import pandas as pd

# ── Step 1: Clean occupation list ─────────────────────────────────────────────
df = pd.read_excel(
    'Download_data/Career Drive Project Data Sources.xlsx',
    sheet_name='Construction ONet Codes',
    header=None,
    dtype=str  # prevent SOC codes being read as dates
)
df.columns = ['soc_code', 'occ_title', 'role_description']
df = df[df['soc_code'].str.match(r'^\d{2}-\d{4}', na=False)].copy()
df['soc_code'] = (df['soc_code']
                  .str.strip()
                  .str.replace('.00', '', regex=False))

df = df.drop_duplicates(subset='soc_code', keep='first')
print(f"{len(df)} unique occupations retained")

# ── Step 2: Load BLS — build Maine + fallback tables ─────────────────────────
bls = pd.read_excel("Download_data/state_M2024_dl.xlsx", dtype={'OCC_CODE': str})
bls['OCC_CODE'] = (bls['OCC_CODE']
                   .str.replace('.00', '', regex=False)
                   .str.replace('.01', '', regex=False)
                   .str.strip())
bls['A_MEDIAN'] = pd.to_numeric(bls['A_MEDIAN'], errors='coerce')

# Fallback 1: Maine exact match
maine = (bls[bls['PRIM_STATE'] == 'ME'][['OCC_CODE', 'A_MEDIAN']]
         .rename(columns={'OCC_CODE': 'soc_code', 'A_MEDIAN': 'wage_maine'}))

# Fallback 2: Maine parent SOC (e.g. 17-2051.01 → 17-2051)
maine_all = bls[bls['PRIM_STATE'] == 'ME'].copy()
maine_all['soc_parent'] = maine_all['OCC_CODE'].str[:7]
maine_parent = (maine_all.groupby('soc_parent')['A_MEDIAN']
                .mean().reset_index()
                .rename(columns={'soc_parent': 'soc_code', 'A_MEDIAN': 'wage_parent'}))

# Fallback 3: New England neighbors average (NH, VT, MA)
ne_avg = (bls[bls['PRIM_STATE'].isin(['NH', 'VT', 'MA'])]
          .groupby('OCC_CODE')['A_MEDIAN']
          .mean().reset_index()
          .rename(columns={'OCC_CODE': 'soc_code', 'A_MEDIAN': 'wage_ne_avg'}))

# Fallback 4: National median
nat_avg = (bls[bls['PRIM_STATE'].isna()]
           .groupby('OCC_CODE')['A_MEDIAN']
           .mean().reset_index()
           .rename(columns={'OCC_CODE': 'soc_code', 'A_MEDIAN': 'wage_national'}))

# ── Step 3: Join all fallbacks ────────────────────────────────────────────────
master = df.merge(maine,        on='soc_code', how='left')
master = master.merge(maine_parent, on='soc_code', how='left')
master = master.merge(ne_avg,   on='soc_code', how='left')
master = master.merge(nat_avg,  on='soc_code', how='left')

# ── Step 4: Auto-fill with priority order ────────────────────────────────────
def fill_wage(row):
    """
    Apply a 4-level wage fallback for a single occupation row.

    Priority order:
        1. Maine exact match         (PRIM_STATE == 'ME')
        2. Maine parent SOC average  (e.g. 17-2051.01 → 17-2051)
        3. New England neighbors avg (NH, VT, MA)
        4. National median           (no state filter)

    Args:
        row (pd.Series): A row from the master DataFrame containing
                         wage_maine, wage_parent, wage_ne_avg, wage_national.

    Returns:
        tuple: (wage_value: float or None, wage_source: str)
    """
    if pd.notna(row['wage_maine']):
        return row['wage_maine'], 'Maine'
    elif pd.notna(row['wage_parent']):
        return row['wage_parent'], 'Maine_parent'
    elif pd.notna(row['wage_ne_avg']):
        return row['wage_ne_avg'], 'NE_neighbor_avg'
    elif pd.notna(row['wage_national']):
        return row['wage_national'], 'National'
    else:
        return None, 'Missing'

master[['median_wage_me', 'wage_source']] = master.apply(
    fill_wage, axis=1, result_type='expand'
)

# ── Step 4b: Patch 17-2051.01 — hard-coded national wage ─────────────────────
master.loc[master['soc_code'] == '17-2051.01', 'median_wage_me'] = 70510
master.loc[master['soc_code'] == '17-2051.01', 'wage_source']    = 'Manual_national'
print("17-2051.01 patched: $70,510")

# ── Step 4c: Patch 47-2171 — continental US average ──────────────────────────
rebar_avg = (bls[
    (bls['OCC_CODE'] == '47-2171') &
    (~bls['PRIM_STATE'].isin(['PR', 'GU', 'VI'])) &
    (bls['A_MEDIAN'].notna())
]['A_MEDIAN'].mean())

master.loc[master['soc_code'] == '47-2171', 'median_wage_me'] = rebar_avg
master.loc[master['soc_code'] == '47-2171', 'wage_source']    = 'Continental_avg'
print(f"47-2171 patched with continental avg: ${rebar_avg:,.0f}")

# ── Step 5: Join O*NET skill descriptions ────────────────────────────────────
skills = pd.read_excel('Download_data/Skills.xlsx', dtype={'O*NET-SOC Code': str})

# Keep only Importance scale (IM), filter to construction-related SOC codes
skills = skills[skills['Scale ID'] == 'IM'].copy()
skills['soc_code'] = (skills['O*NET-SOC Code']
                      .str.replace('.00', '', regex=False)
                      .str.strip())

# Combine all skill names per occupation into one text string
skill_text = (skills.groupby('soc_code')['Element Name']
              .apply(lambda x: ', '.join(x.dropna().unique()))
              .reset_index()
              .rename(columns={'Element Name': 'skill_description'}))

# First pass: exact match (handles 17-2051.01, 13-1082.00 → 13-1082)
master = master.merge(skill_text, on='soc_code', how='left')

# Second pass: fallback using parent code (e.g. 17-2051 matches 17-2051.01 skills)
skill_text['soc_parent'] = skill_text['soc_code'].str[:7]
skill_parent = (skill_text.groupby('soc_parent')['skill_description']
                .apply(lambda x: ', '.join(x.dropna().unique()))
                .reset_index()
                .rename(columns={'soc_parent': 'soc_code',
                                 'skill_description': 'skill_parent'}))

master = master.merge(skill_parent, on='soc_code', how='left')
master['skill_description'] = master['skill_description'].fillna(master['skill_parent'])
master = master.drop(columns=['skill_parent'])

# ── Step 5c: Patch 13-1082 — manual skill description ────────────────────────
master.loc[master['soc_code'] == '13-1082', 'skill_description'] = (
    "Project Management, Coordination, Time Management, Critical Thinking, "
    "Active Listening, Speaking, Writing, Judgment and Decision Making, "
    "Monitoring, Management of Personnel Resources, Management of Material Resources, "
    "Management of Financial Resources, Complex Problem Solving, Negotiation"
)
print("13-1082 skill_description patched manually")

# ── Step 5d: Load Task Statements and merge ───────────────────────────────────
tasks = pd.read_excel('Download_data/Task Statements.xlsx',
                      dtype={'O*NET-SOC Code': str})

print("── Task Statements Raw Head ──")
print(tasks.head(3).to_string())
print(f"Task Type values: {tasks['Task Type'].unique()}")
print()

tasks['soc_code'] = (tasks['O*NET-SOC Code']
                     .str.replace('.00', '', regex=False)
                     .str.strip())

# Only keep Core tasks (drop Supplemental to reduce noise)
tasks = tasks[tasks['Task Type'] == 'Core']

# Combine all tasks per occupation into one text string
task_text = (tasks.groupby('soc_code')['Task']
             .apply(lambda x: ', '.join(x.dropna().unique()))
             .reset_index()
             .rename(columns={'Task': 'task_description'}))

master = master.merge(task_text, on='soc_code', how='left')

# Combine Skills + Tasks into final skill_description
master['skill_description'] = (
    'Tasks: ' + master['task_description'].fillna('') +
    '. Skills: ' + master['skill_description'].fillna('')
)
master = master.drop(columns=['task_description'])

print(f"\n── Task descriptions joined ──")
print(f"Missing tasks: {master['skill_description'].isna().sum()}")

# Check for missing skill descriptions
missing_skills = master[master['skill_description'].isna()]
print(f"\n── Skills Missing: {len(missing_skills)} occupations ──")
if len(missing_skills) > 0:
    print(missing_skills[['soc_code', 'occ_title']])

# ── Step 5b: Drop intermediate columns, save ─────────────────────────────────
master = master.drop(columns=['wage_maine', 'wage_parent', 'wage_ne_avg', 'wage_national'])
master.to_csv('occupations_master.csv', index=False)

# ── Step 6: Audit report ─────────────────────────────────────────────────────
print("\n── Wage Source Summary ──")
print(master['wage_source'].value_counts().to_string())

print("\n── Still Missing ──")
still_missing = master[master['wage_source'] == 'Missing']
if len(still_missing) == 0:
    print("None — all occupations have a wage ✓")
else:
    print(still_missing[['soc_code', 'occ_title']])

print("\n── Sample skill_description (Electricians) ──")
print(master.loc[master['occ_title'] == 'Electricians', 'skill_description'].values[0])

print("\n── skill_description length stats ──")
master['desc_len'] = master['skill_description'].str.len()
print(master[['occ_title', 'desc_len']].to_string())
master = master.drop(columns=['desc_len'])