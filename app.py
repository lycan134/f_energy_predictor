import streamlit as st
import pandas as pd
from backend.predict_energy import predict_energy
from backend.featurize import featurize_formula_with_spacegroups  # vectorized featurizer

# --- Streamlit Page Config ---
st.set_page_config(page_title="Formation Energy Predictor", layout="wide")
st.title("🔬 Formation Energy Prediction App")
st.write(
    "Predict formation energy per atom across all 230 space groups for a given chemical formula."
)

# --- Input Section ---
formula = st.text_input("Enter Chemical Formula (e.g., NiO, MgNiO2):")

if st.button("Featurize Formula"):
    if not formula:
        st.warning("⚠️ Please enter a valid chemical formula first.")
    else:
        with st.spinner("Featurizing formula for all 230 spacegroups..."):
            try:
                # Vectorized featurization for all 300 spacegroups
                df_features = featurize_formula_with_spacegroups(formula)

                st.success("✅ Featurization complete!")
                st.write(f"**Shape:** {df_features.shape} (300 rows × 415 columns)")
                st.dataframe(df_features.head(10), use_container_width=True)

                # Download button for features CSV
                csv = df_features.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Featurized CSV",
                    data=csv,
                    file_name=f"{formula}_featurized_230_spacegroups.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Error during featurization: {e}")

# --- Prediction Section ---
if st.button("Predict Formation Energy"):
    if not formula:
        st.warning("⚠️ Please enter a valid chemical formula first.")
    else:
        with st.spinner("Featurizing and predicting formation energies for all 230 spacegroups..."):
            try:
                # Step 1: Featurize (vectorized)
                df_features = featurize_formula_with_spacegroups(formula)

                # Step 2: Predict formation energy (batch)
                preds = predict_energy(df_features)

                # Step 3: Combine results
                df_results = pd.DataFrame({
                    "Chemical Formula": [formula] * len(preds),
                    "Space Group": range(1, len(preds) + 1),
                    "Predicted Formation Energy (eV/atom)": preds
                })

                # Step 4: Display results
                st.success("✅ Prediction Complete!")
                st.dataframe(df_results, use_container_width=True)

                # Download button for predictions CSV
                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name=f"{formula}_formation_energy_230_spacegroups.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
