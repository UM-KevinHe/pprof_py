lre_model.plot_funnel(
    stdz='indirect',
    null='median',       # Baseline for the difference
    target=0.0,          # Target line for the difference
    alpha=[0.05, 0.01],  # For 95% and 99% control limits
    plot_title="Funnel Plot: Standardized Differences vs Provider Size",
    xlab="Provider Size (Number of Patients)",
    ylab="Standardized Difference (based on Random Effects)"
)