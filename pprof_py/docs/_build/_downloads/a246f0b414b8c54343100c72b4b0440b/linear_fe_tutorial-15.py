lfe_model.plot_funnel(
    stdz='indirect',
    null='median',
    target=0.0,
    alpha=[0.05, 0.01],
    plot_title="Funnel Plot: Standardized LOS Differences vs Provider Size",
    xlab="Provider Size (Number of Patients)",
    ylab="Standardized Difference in LOS"
)