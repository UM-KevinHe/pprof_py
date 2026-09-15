lre_model.plot_provider_effects(
    level=0.95,
    use_flags=True,
    null='median', # Baseline for flagging random effects
    plot_title="Provider Performance: Adjusted Random Effects (u_i)",
    figure_size=(8, 6)
)