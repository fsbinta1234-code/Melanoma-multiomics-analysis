"""
Phase 3 — Temporal kinase activation dynamics (demonstration).

NOTE: the available dataset has no time series (6h/24h/48h/72h); the values below
are ILLUSTRATIVE, following the pipeline's teaching material, to demonstrate the
expected adaptive-rewiring pattern (acute ERK → late AKT/mTOR) during the
acquisition of BRAFi/MEKi resistance.
"""
import matplotlib.pyplot as plt

import pipeline_config as cfg


class TemporalKinaseActivationDynamics:

    TIME_POINTS = ["6h", "24h", "48h", "72h"]
    CURVES = {
        "ERK": [2.1, 1.4, 2.7, 4.2],
        "AKT": [1.0, 1.8, 3.0, 4.1],
        "mTOR": [0.9, 1.7, 2.8, 3.9],
    }
    CURVE_COLORS = {"ERK": cfg.COLOR_RESIST, "AKT": cfg.COLOR_CONTROL, "mTOR": cfg.COLOR_ACCENT}

    @staticmethod
    def run() -> None:
        cfg.apply_style()
        fig, ax = plt.subplots(figsize=(9, 6))
        for name, values in TemporalKinaseActivationDynamics.CURVES.items():
            ax.plot(TemporalKinaseActivationDynamics.TIME_POINTS, values, marker="o",
                    linewidth=3, markersize=9, label=name,
                    color=TemporalKinaseActivationDynamics.CURVE_COLORS[name])
        ax.set_xlabel("Treatment time")
        ax.set_ylabel("Kinase activation score (log2)")
        ax.set_title("Temporal Kinase Activation Dynamics (illustrative)\n"
                     "BRAFi/MEKi Resistance")
        ax.legend(title="Kinase")
        cfg.save_figure(fig, "03_temporal_kinase_activation")
        print("Temporal figure (illustrative) saved.")


def main():
    TemporalKinaseActivationDynamics.run()


if __name__ == "__main__":
    main()
