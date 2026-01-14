import csv
import io
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "Code Société",
    "No facture",
    "Code Tiers",
    "Raison sociale",
    "Libellé écriture",
    "Type de pièce",
    "Date facture",
    "Date d'échéance",
    "Montant Signé",
    "Devise comptabilisation",
    "Code du compte général",
    "Numéro d'écriture",
]

ACCOUNT_FILTER = "41100000"
MAX_GROUP_SIZE = 20
MAX_SOLUTIONS_PER_GROUP = 50
GROUP_TIME_LIMIT_S = 1.0


@dataclass
class MatchSolution:
    code_tiers: str
    raison_sociale: str
    mois_echeance: str
    ecritures: list
    factures: list
    montants: list
    dates_echeance: list
    somme: int
    fiable: bool


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    raw = uploaded_file.read()
    if not raw:
        raise ValueError("Fichier vide.")

    encoding_used = None
    sample_text = None
    for encoding in ["latin1", "cp1252", "utf-8-sig"]:
        try:
            sample_text = raw[:4096].decode(encoding)
            encoding_used = encoding
            break
        except UnicodeDecodeError:
            continue

    if encoding_used is None:
        raise ValueError("Encodage non supporté.")

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=";,")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ";"

    buffer = io.BytesIO(raw)
    return pd.read_csv(
        buffer,
        sep=delimiter,
        encoding=encoding_used,
        dtype={"Montant Signé": "string"},
    )


def parse_amount_to_cents(value: str) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Montant manquant")
    normalized = str(value).strip().replace(" ", "").replace(",", ".")
    try:
        dec_value = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"Montant invalide: {value}") from exc
    return int(dec_value * 100)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes: " + ", ".join(missing)
        )

    df = df.copy()
    df = df[df["Code du compte général"].astype(str) == ACCOUNT_FILTER]

    df["Date d'échéance"] = pd.to_datetime(
        df["Date d'échéance"],
        errors="coerce",
        dayfirst=True,
    )
    if df["Date d'échéance"].isna().any():
        raise ValueError("Dates d'échéance non parseables détectées.")

    today = pd.Timestamp.now(tz="Europe/Paris").normalize().tz_localize(None)
    df = df[df["Date d'échéance"].dt.date <= today.date()]

    df["montant_int"] = df["Montant Signé"].apply(parse_amount_to_cents)
    df["mois_echeance"] = df["Date d'échéance"].dt.to_period("M").astype(str)
    df["Code Tiers"] = df["Code Tiers"].astype(str)
    df["Raison sociale"] = df["Raison sociale"].astype(str)
    return df


def _pair_matches(indices, amounts):
    by_amount = defaultdict(list)
    for idx in indices:
        by_amount[amounts[idx]].append(idx)

    used = set()
    solutions = []
    for amount, idx_list in list(by_amount.items()):
        if amount <= 0:
            continue
        opposite = -amount
        if opposite not in by_amount:
            continue
        while idx_list and by_amount[opposite]:
            i = idx_list.pop()
            j = by_amount[opposite].pop()
            if i in used or j in used:
                continue
            used.update({i, j})
            solutions.append([i, j])
    return solutions, used


def _generate_subsets(indices, amounts, max_size, deadline):
    subsets = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(indices, size):
            if time.monotonic() > deadline:
                return subsets
            total = sum(amounts[i] for i in combo)
            subsets.append((total, combo))
    return subsets


def _find_combo_solutions(indices, amounts, max_solutions, time_limit_s):
    deadline = time.monotonic() + time_limit_s
    if len(indices) < 3:
        return []

    left = indices[: len(indices) // 2]
    right = indices[len(indices) // 2 :]

    left_subsets = _generate_subsets(left, amounts, 3, deadline)
    right_subsets = _generate_subsets(right, amounts, 3, deadline)

    sum_to_right = defaultdict(list)
    for total, combo in right_subsets:
        sum_to_right[total].append(combo)

    solutions = []

    for total, combo in left_subsets:
        if time.monotonic() > deadline or len(solutions) >= max_solutions:
            break
        if total == 0 and 3 <= len(combo) <= 6:
            solutions.append(list(combo))
            if len(solutions) >= max_solutions:
                break
        complement = -total
        for right_combo in sum_to_right.get(complement, []):
            if time.monotonic() > deadline or len(solutions) >= max_solutions:
                break
            merged = combo + right_combo
            if 3 <= len(merged) <= 6:
                solutions.append(list(merged))

    for total, combo in right_subsets:
        if time.monotonic() > deadline or len(solutions) >= max_solutions:
            break
        if total == 0 and 3 <= len(combo) <= 6:
            solutions.append(list(combo))

    return solutions


def find_matches_for_group(df_group: pd.DataFrame):
    amounts = df_group["montant_int"].to_dict()
    indices = list(df_group.index)

    negatives = [amounts[i] for i in indices if amounts[i] < 0]
    positives = [amounts[i] for i in indices if amounts[i] > 0]
    if not negatives:
        return [], "pas de négatif"
    if positives and min(positives) > abs(sum(negatives)):
        return [], "filtre min positif"

    solutions_indices = []
    pair_solutions, used = _pair_matches(indices, amounts)
    solutions_indices.extend(pair_solutions)

    remaining = [idx for idx in indices if idx not in used]
    if remaining:
        combo_solutions = _find_combo_solutions(
            remaining,
            amounts,
            MAX_SOLUTIONS_PER_GROUP - len(solutions_indices),
            GROUP_TIME_LIMIT_S,
        )
        for combo in combo_solutions:
            if any(idx in used for idx in combo):
                continue
            used.update(combo)
            solutions_indices.append(combo)
            if len(solutions_indices) >= MAX_SOLUTIONS_PER_GROUP:
                break

    if not solutions_indices:
        return [], "aucune combinaison"
    return solutions_indices, "ok"


def build_solution_rows(df_group: pd.DataFrame, combos, mois_echeance: str):
    rows = []
    for combo in combos:
        slice_df = df_group.loc[list(combo)]
        fiable = slice_df["Type de pièce"].astype(str).str.upper().eq("RC").any()
        rows.append(
            MatchSolution(
                code_tiers=str(slice_df["Code Tiers"].iloc[0]),
                raison_sociale=str(slice_df["Raison sociale"].iloc[0]),
                mois_echeance=mois_echeance,
                ecritures=slice_df["Numéro d'écriture"].astype(str).tolist(),
                factures=slice_df["No facture"].astype(str).tolist(),
                montants=slice_df["Montant Signé"].astype(str).tolist(),
                dates_echeance=slice_df["Date d'échéance"].dt.strftime("%Y-%m-%d").tolist(),
                somme=int(slice_df["montant_int"].sum()),
                fiable=fiable,
            )
        )
    return rows


def main():
    st.set_page_config(page_title="Lettrage exact", layout="wide")
    st.title("Détection de lettrages exacts")
    st.markdown(
        "Chargez une balance clients non-lettrée. L'application recherche des combinaisons "
        "de montants qui s'annulent exactement (somme = 0) par code tiers."
    )

    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    if not uploaded_file:
        st.stop()

    start_time = time.monotonic()

    try:
        df = load_csv(uploaded_file)
        df_filtered = preprocess(df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    stats = {
        "lignes_initiales": len(df),
        "lignes_filtrees": len(df_filtered),
    }

    results = []

    grouped = df_filtered.groupby("Code Tiers", sort=False)
    progress = st.progress(0)

    for idx, (code_tiers, group_df) in enumerate(grouped, start=1):
        progress.progress(idx / len(grouped))
        if len(group_df) > MAX_GROUP_SIZE:
            sub_groups = group_df.groupby("mois_echeance", sort=False)
        else:
            sub_groups = [("global", group_df)]

        for mois_echeance, sub_df in sub_groups:
            combos, reason = find_matches_for_group(sub_df)
            if reason == "ok":
                results.extend(build_solution_rows(sub_df, combos, mois_echeance))

    elapsed = time.monotonic() - start_time
    stats.update(
        {
            "codes_tiers": df_filtered["Code Tiers"].nunique(),
            "codes_tiers_avec_solution": len({r.code_tiers for r in results}),
            "temps_calcul_s": round(elapsed, 2),
        }
    )

    st.subheader("Statistiques")
    st.json(stats)

    st.subheader("Lettrages trouvés")
    if results:
        df_results = pd.DataFrame(
            [
                {
                    "Code Tiers": r.code_tiers,
                    "Raison sociale": r.raison_sociale,
                    "Mois d'échéance": r.mois_echeance,
                    "Numéros d'écriture": " | ".join(r.ecritures),
                    "No facture": " | ".join(r.factures),
                    "Nb pièces": len(r.ecritures),
                    "Somme (centimes)": r.somme,
                    "Montants": " | ".join(r.montants),
                    "Dates d'échéance": " | ".join(r.dates_echeance),
                    "Fiable": "Oui" if r.fiable else "À vérifier",
                }
                for r in results
            ]
        )
        df_fiables = df_results[df_results["Fiable"] == "Oui"].copy()
        df_a_verifier = df_results[df_results["Fiable"] == "À vérifier"].copy()

        st.markdown("### Lettrages fiables (avec règlement RC)")
        if not df_fiables.empty:
            st.dataframe(df_fiables, use_container_width=True)
            csv_bytes = df_fiables.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les lettrages fiables (CSV)",
                data=csv_bytes,
                file_name="lettrages_fiables.csv",
                mime="text/csv",
            )
        else:
            st.info("Aucun lettrage fiable trouvé.")

        st.markdown("### Lettrages à vérifier (sans RC)")
        if not df_a_verifier.empty:
            st.dataframe(df_a_verifier, use_container_width=True)
            csv_bytes = df_a_verifier.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les lettrages à vérifier (CSV)",
                data=csv_bytes,
                file_name="lettrages_a_verifier.csv",
                mime="text/csv",
            )
        else:
            st.info("Aucun lettrage à vérifier trouvé.")
    else:
        st.info("Aucun lettrage trouvé.")

    st.subheader("Commandes de lancement")
    st.code("pip install -r requirements.txt\nstreamlit run app.py")


if __name__ == "__main__":
    main()
