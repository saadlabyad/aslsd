Search.setIndex({docnames:["aslsd","aslsd.applications","aslsd.basis_kernels","aslsd.basis_kernels.tests","aslsd.estimators","aslsd.estimators.tests","aslsd.evaluation","aslsd.evaluation.tests","aslsd.events","aslsd.events.tests","aslsd.kernels","aslsd.models","aslsd.optim_logging","aslsd.plots","aslsd.solvers","aslsd.utilities","index","modules","readme"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["aslsd.rst","aslsd.applications.rst","aslsd.basis_kernels.rst","aslsd.basis_kernels.tests.rst","aslsd.estimators.rst","aslsd.estimators.tests.rst","aslsd.evaluation.rst","aslsd.evaluation.tests.rst","aslsd.events.rst","aslsd.events.tests.rst","aslsd.kernels.rst","aslsd.models.rst","aslsd.optim_logging.rst","aslsd.plots.rst","aslsd.solvers.rst","aslsd.utilities.rst","index.rst","modules.rst","readme.rst"],objects:{"":[[0,0,0,"-","aslsd"]],"aslsd.basis_kernels":[[2,0,0,"-","basis_kernel"],[2,0,0,"-","basis_kernel_delayedexp"],[2,0,0,"-","basis_kernel_exponential"],[2,0,0,"-","basis_kernel_gaussian"],[2,0,0,"-","basis_kernel_null"],[2,0,0,"-","basis_kernel_rayleigh"],[2,0,0,"-","basis_kernel_triangle"],[3,0,0,"-","tests"]],"aslsd.basis_kernels.basis_kernel":[[2,1,1,"","BasisKernel"]],"aslsd.basis_kernels.basis_kernel.BasisKernel":[[2,2,1,"","allow_simu"],[2,2,1,"","dict_interactions"],[2,2,1,"","diff_cross_upsilon"],[2,2,1,"","diff_kl_divergence"],[2,2,1,"","diff_l1_norm"],[2,2,1,"","diff_phi"],[2,2,1,"","diff_psi"],[2,2,1,"","diff_sim_upsilon"],[2,3,1,"","get_interactions"],[2,3,1,"","get_n_param"],[2,3,1,"","get_n_vars"],[2,3,1,"","get_param_bounds"],[2,3,1,"","get_param_names"],[2,3,1,"","get_var_bounds"],[2,3,1,"","get_var_names"],[2,3,1,"","has_omega_param"],[2,3,1,"","is_compatible"],[2,2,1,"","ix_map"],[2,2,1,"","kl_divergence"],[2,2,1,"","l1_norm"],[2,2,1,"","l2_distance"],[2,2,1,"","l2_dot"],[2,2,1,"","l2_norm"],[2,3,1,"","make_dict_interactions"],[2,3,1,"","make_diff_cross_upsilon"],[2,3,1,"","make_diff_cross_upsilon_rev"],[2,3,1,"","make_diff_kl_divergence"],[2,3,1,"","make_diff_kl_divergence_rev"],[2,3,1,"","make_diff_l1_norm"],[2,3,1,"","make_diff_l2_dot"],[2,3,1,"","make_diff_l2_norm"],[2,3,1,"","make_diff_phi"],[2,3,1,"","make_diff_psi"],[2,3,1,"","make_diff_sim_upsilon"],[2,3,1,"","make_ix_map"],[2,3,1,"","make_kernel_functionals"],[2,3,1,"","make_kl_divergence"],[2,3,1,"","make_kl_divergence_rev"],[2,3,1,"","make_kl_functionals"],[2,3,1,"","make_l1_metrics"],[2,3,1,"","make_l1_norm"],[2,3,1,"","make_l2_dot"],[2,3,1,"","make_l2_metrics"],[2,3,1,"","make_l2_norm"],[2,3,1,"","make_phi"],[2,3,1,"","make_psi"],[2,3,1,"","make_simu_func"],[2,3,1,"","make_upsilon"],[2,3,1,"","make_upsilon_rev"],[2,3,1,"","make_vars"],[2,2,1,"","n_fixed_vars"],[2,2,1,"","phi"],[2,2,1,"","psi"],[2,2,1,"","sim_func"],[2,2,1,"","upsilon"]],"aslsd.basis_kernels.basis_kernel_delayedexp":[[2,1,1,"","DelayedExponentialKernel"],[2,4,1,"","diff_cross_upsilon_delayedexp_delayedexp"],[2,4,1,"","diff_cross_upsilon_delayedexp_null"],[2,4,1,"","diff_l2_dot_delayedexp_delayedexp"],[2,4,1,"","diff_l2_dot_delayedexp_null"],[2,4,1,"","l2_dot_delayedexp_delayedexp"],[2,4,1,"","l2_dot_delayedexp_null"],[2,4,1,"","upsilon_delayedexp_delayedexp"],[2,4,1,"","upsilon_delayedexp_null"]],"aslsd.basis_kernels.basis_kernel_delayedexp.DelayedExponentialKernel":[[2,3,1,"","get_interactions"],[2,3,1,"","get_n_vars"],[2,3,1,"","get_var_bounds"],[2,3,1,"","get_var_names"],[2,3,1,"","make_diff_cross_upsilon"],[2,3,1,"","make_diff_cross_upsilon_rev"],[2,3,1,"","make_diff_kl_divergence"],[2,3,1,"","make_diff_kl_divergence_rev"],[2,3,1,"","make_diff_l1_norm"],[2,3,1,"","make_diff_l2_dot"],[2,3,1,"","make_diff_l2_norm"],[2,3,1,"","make_diff_phi"],[2,3,1,"","make_diff_psi"],[2,3,1,"","make_diff_sim_upsilon"],[2,3,1,"","make_kl_divergence"],[2,3,1,"","make_kl_divergence_rev"],[2,3,1,"","make_l1_norm"],[2,3,1,"","make_l2_dot"],[2,3,1,"","make_l2_norm"],[2,3,1,"","make_phi"],[2,3,1,"","make_psi"],[2,3,1,"","make_simu_func"],[2,3,1,"","make_upsilon"],[2,3,1,"","make_upsilon_rev"]],"aslsd.basis_kernels.basis_kernel_exponential":[[2,1,1,"","ExponentialKernel"],[2,4,1,"","diff_cross_upsilon_exp_exp"],[2,4,1,"","diff_cross_upsilon_exp_gauss"],[2,4,1,"","diff_cross_upsilon_exp_null"],[2,4,1,"","diff_l2_dot_exp_exp"],[2,4,1,"","diff_l2_dot_exp_gauss"],[2,4,1,"","diff_l2_dot_exp_null"],[2,4,1,"","l2_dot_exp_exp"],[2,4,1,"","l2_dot_exp_gauss"],[2,4,1,"","l2_dot_exp_null"],[2,4,1,"","upsilon_exponential_exponential"],[2,4,1,"","upsilon_exponential_gaussian"],[2,4,1,"","upsilon_exponential_null"]],"aslsd.basis_kernels.basis_kernel_exponential.ExponentialKernel":[[2,3,1,"","get_interactions"],[2,3,1,"","get_n_vars"],[2,3,1,"","get_var_bounds"],[2,3,1,"","get_var_names"],[2,3,1,"","make_diff_cross_upsilon"],[2,3,1,"","make_diff_cross_upsilon_rev"],[2,3,1,"","make_diff_kl_divergence"],[2,3,1,"","make_diff_kl_divergence_rev"],[2,3,1,"","make_diff_l1_norm"],[2,3,1,"","make_diff_l2_dot"],[2,3,1,"","make_diff_l2_norm"],[2,3,1,"","make_diff_phi"],[2,3,1,"","make_diff_psi"],[2,3,1,"","make_diff_sim_upsilon"],[2,3,1,"","make_kl_divergence"],[2,3,1,"","make_kl_divergence_rev"],[2,3,1,"","make_l1_norm"],[2,3,1,"","make_l2_dot"],[2,3,1,"","make_l2_norm"],[2,3,1,"","make_phi"],[2,3,1,"","make_psi"],[2,3,1,"","make_simu_func"],[2,3,1,"","make_upsilon"],[2,3,1,"","make_upsilon_rev"]],"aslsd.basis_kernels.basis_kernel_gaussian":[[2,1,1,"","GaussianKernel"],[2,4,1,"","diff_cross_upsilon_gauss_exp"],[2,4,1,"","diff_cross_upsilon_gauss_gauss"],[2,4,1,"","diff_cross_upsilon_gauss_null"],[2,4,1,"","diff_kl_divergence_gauss_gauss"],[2,4,1,"","diff_l2_dot_gauss_exp"],[2,4,1,"","diff_l2_dot_gauss_gauss"],[2,4,1,"","diff_l2_dot_gauss_null"],[2,4,1,"","kl_divergence_gauss_gauss"],[2,4,1,"","l2_dot_gauss_exp"],[2,4,1,"","l2_dot_gauss_gauss"],[2,4,1,"","l2_dot_gauss_null"],[2,4,1,"","upsilon_gauss_exp"],[2,4,1,"","upsilon_gauss_gauss"],[2,4,1,"","upsilon_gauss_null"]],"aslsd.basis_kernels.basis_kernel_gaussian.GaussianKernel":[[2,3,1,"","get_interactions"],[2,3,1,"","get_n_vars"],[2,3,1,"","get_var_bounds"],[2,3,1,"","get_var_names"],[2,3,1,"","make_diff_cross_upsilon"],[2,3,1,"","make_diff_cross_upsilon_rev"],[2,3,1,"","make_diff_kl_divergence"],[2,3,1,"","make_diff_kl_divergence_rev"],[2,3,1,"","make_diff_l1_norm"],[2,3,1,"","make_diff_l2_dot"],[2,3,1,"","make_diff_l2_norm"],[2,3,1,"","make_diff_phi"],[2,3,1,"","make_diff_psi"],[2,3,1,"","make_diff_sim_upsilon"],[2,3,1,"","make_kl_divergence"],[2,3,1,"","make_kl_divergence_rev"],[2,3,1,"","make_l1_norm"],[2,3,1,"","make_l2_dot"],[2,3,1,"","make_l2_norm"],[2,3,1,"","make_phi"],[2,3,1,"","make_psi"],[2,3,1,"","make_simu_func"],[2,3,1,"","make_upsilon"],[2,3,1,"","make_upsilon_rev"]],"aslsd.basis_kernels.basis_kernel_null":[[2,1,1,"","NullKernel"]],"aslsd.basis_kernels.basis_kernel_null.NullKernel":[[2,3,1,"","get_interactions"],[2,3,1,"","get_n_vars"],[2,3,1,"","get_var_bounds"],[2,3,1,"","get_var_names"],[2,3,1,"","make_diff_cross_upsilon"],[2,3,1,"","make_diff_cross_upsilon_rev"],[2,3,1,"","make_diff_kl_divergence"],[2,3,1,"","make_diff_kl_divergence_rev"],[2,3,1,"","make_diff_l1_norm"],[2,3,1,"","make_diff_l2_norm"],[2,3,1,"","make_diff_phi"],[2,3,1,"","make_diff_psi"],[2,3,1,"","make_diff_sim_upsilon"],[2,3,1,"","make_kl_divergence"],[2,3,1,"","make_kl_divergence_rev"],[2,3,1,"","make_l1_norm"],[2,3,1,"","make_l2_dot"],[2,3,1,"","make_l2_norm"],[2,3,1,"","make_phi"],[2,3,1,"","make_psi"],[2,3,1,"","make_simu_func"],[2,3,1,"","make_upsilon"],[2,3,1,"","make_upsilon_rev"]],"aslsd.basis_kernels.basis_kernel_rayleigh":[[2,1,1,"","RayleighKernel"]],"aslsd.basis_kernels.basis_kernel_rayleigh.RayleighKernel":[[2,3,1,"","get_interactions"],[2,3,1,"","get_n_vars"],[2,3,1,"","get_var_bounds"],[2,3,1,"","get_var_names"],[2,3,1,"","make_diff_cross_upsilon"],[2,3,1,"","make_diff_cross_upsilon_rev"],[2,3,1,"","make_diff_kl_divergence"],[2,3,1,"","make_diff_kl_divergence_rev"],[2,3,1,"","make_diff_l1_norm"],[2,3,1,"","make_diff_l2_dot"],[2,3,1,"","make_diff_l2_norm"],[2,3,1,"","make_diff_phi"],[2,3,1,"","make_diff_psi"],[2,3,1,"","make_diff_sim_upsilon"],[2,3,1,"","make_kl_divergence"],[2,3,1,"","make_kl_divergence_rev"],[2,3,1,"","make_l1_norm"],[2,3,1,"","make_l2_dot"],[2,3,1,"","make_l2_norm"],[2,3,1,"","make_phi"],[2,3,1,"","make_psi"],[2,3,1,"","make_simu_func"],[2,3,1,"","make_upsilon"],[2,3,1,"","make_upsilon_rev"]],"aslsd.basis_kernels.basis_kernel_triangle":[[2,1,1,"","TriangularKernel"]],"aslsd.basis_kernels.basis_kernel_triangle.TriangularKernel":[[2,3,1,"","get_interactions"],[2,3,1,"","get_n_vars"],[2,3,1,"","get_var_bounds"],[2,3,1,"","get_var_names"],[2,3,1,"","make_diff_cross_upsilon"],[2,3,1,"","make_diff_cross_upsilon_rev"],[2,3,1,"","make_diff_kl_divergence"],[2,3,1,"","make_diff_kl_divergence_rev"],[2,3,1,"","make_diff_l1_norm"],[2,3,1,"","make_diff_l2_dot"],[2,3,1,"","make_diff_l2_norm"],[2,3,1,"","make_diff_phi"],[2,3,1,"","make_diff_psi"],[2,3,1,"","make_diff_sim_upsilon"],[2,3,1,"","make_kl_divergence"],[2,3,1,"","make_kl_divergence_rev"],[2,3,1,"","make_l1_norm"],[2,3,1,"","make_l2_dot"],[2,3,1,"","make_l2_norm"],[2,3,1,"","make_phi"],[2,3,1,"","make_psi"],[2,3,1,"","make_simu_func"],[2,3,1,"","make_upsilon"],[2,3,1,"","make_upsilon_rev"]],"aslsd.basis_kernels.tests":[[3,0,0,"-","test"]],"aslsd.estimators":[[4,0,0,"-","adaptive_stratified_estimator"],[4,0,0,"-","double_ix_stratification"],[4,0,0,"-","estimator"],[4,0,0,"-","exact_estimator"],[4,0,0,"-","single_ix_stratification"],[5,0,0,"-","tests"]],"aslsd.estimators.adaptive_stratified_estimator":[[4,1,1,"","AdaptiveStratified"]],"aslsd.estimators.adaptive_stratified_estimator.AdaptiveStratified":[[4,3,1,"","diff_lse_mu_k"],[4,3,1,"","diff_lse_theta_kpr"],[4,3,1,"","estimate_dphi"],[4,3,1,"","estimate_dphi_adaptivedom"],[4,3,1,"","estimate_dphi_adaptivedom_gtarget"],[4,3,1,"","estimate_dphi_gtarget"],[4,3,1,"","estimate_dpsi"],[4,3,1,"","estimate_dupsilon"],[4,3,1,"","estimate_dupsilon_adaptivedom"],[4,3,1,"","estimate_dupsilon_adaptivedom_gtarget"],[4,3,1,"","estimate_dupsilon_gtarget"],[4,3,1,"","estimate_dupsilonzero"],[4,3,1,"","get_log"],[4,3,1,"","get_true_allocation_phi"],[4,3,1,"","get_true_allocation_upsilon"],[4,3,1,"","intialize_logs"],[4,3,1,"","lse_k_estimate"],[4,3,1,"","lse_k_grad_estimate"],[4,3,1,"","set_doublesum_estimation"],[4,3,1,"","set_singlesum_estimation"],[4,3,1,"","set_stratification"]],"aslsd.estimators.double_ix_stratification":[[4,1,1,"","DoubleIxStratification"]],"aslsd.estimators.double_ix_stratification.DoubleIxStratification":[[4,3,1,"","get_default_adaptive_rel_alloc_0"],[4,3,1,"","get_default_adaptive_strata"],[4,3,1,"","get_default_nonadaptive_strata"],[4,3,1,"","get_lag_strata_finitepop"],[4,3,1,"","get_lag_strata_sizes"],[4,3,1,"","get_stratification_size"]],"aslsd.estimators.estimator":[[4,1,1,"","Estimator"]],"aslsd.estimators.estimator.Estimator":[[4,3,1,"","get_log"],[4,3,1,"","initialize"],[4,3,1,"","lse_k_estimate"],[4,3,1,"","lse_k_grad_estimate"]],"aslsd.estimators.exact_estimator":[[4,1,1,"","ExactEstimator"]],"aslsd.estimators.exact_estimator.ExactEstimator":[[4,3,1,"","diff_lse_mu_k"],[4,3,1,"","diff_lse_theta_kpr"],[4,3,1,"","get_log"],[4,3,1,"","lse_k_estimate"],[4,3,1,"","lse_k_grad_estimate"]],"aslsd.estimators.single_ix_stratification":[[4,1,1,"","SingleIxStratification"]],"aslsd.estimators.single_ix_stratification.SingleIxStratification":[[4,3,1,"","get_default_abs_alloc"],[4,3,1,"","get_default_strata"],[4,3,1,"","get_stratification_size"]],"aslsd.estimators.tests":[[5,0,0,"-","test"]],"aslsd.evaluation":[[6,0,0,"-","goodness_of_fit"],[7,0,0,"-","tests"]],"aslsd.evaluation.goodness_of_fit":[[6,4,1,"","get_residuals"],[6,4,1,"","get_residuals_hpoisson"],[6,4,1,"","get_residuals_k"],[6,4,1,"","get_residuals_k_hpoisson"],[6,4,1,"","ks_test_residuals"],[6,4,1,"","qq_plot"]],"aslsd.evaluation.tests":[[7,0,0,"-","test"]],"aslsd.events":[[8,0,0,"-","process_path"],[9,0,0,"-","tests"],[8,0,0,"-","time_ordering"]],"aslsd.events.process_path":[[8,1,1,"","ProcessPath"]],"aslsd.events.process_path.ProcessPath":[[8,2,1,"","T_f"],[8,2,1,"","d"],[8,2,1,"","eta"],[8,2,1,"","kappa"],[8,2,1,"","lag_sizes"],[8,2,1,"","list_times"],[8,2,1,"","list_times2end"],[8,2,1,"","n_events"],[8,2,1,"","varpi"]],"aslsd.events.tests":[[9,0,0,"-","time_ordering_test"]],"aslsd.events.tests.time_ordering_test":[[9,4,1,"","test_kappa"],[9,4,1,"","test_varkappa"]],"aslsd.events.time_ordering":[[8,4,1,"","get_kappa_varpi"],[8,4,1,"","get_lag_sizes"]],"aslsd.kernels":[[10,0,0,"-","kernel"]],"aslsd.kernels.kernel":[[10,1,1,"","KernelModel"]],"aslsd.kernels.kernel.KernelModel":[[10,3,1,"","append"],[10,5,1,"","basis_kernels"],[10,3,1,"","diff_l2_projection_loss"],[10,3,1,"","get_l2_projection"],[10,3,1,"","get_l2_projection_loss"],[10,3,1,"","get_param_bounds"],[10,3,1,"","get_vec_param_names"],[10,3,1,"","is_compatible"],[10,3,1,"","is_sbf"],[10,3,1,"","ix_omegas"],[10,3,1,"","make_diff_cross_upsilon"],[10,3,1,"","make_diff_l1_norm"],[10,3,1,"","make_diff_l2_dot"],[10,3,1,"","make_diff_l2_norm"],[10,3,1,"","make_diff_phi"],[10,3,1,"","make_diff_psi"],[10,3,1,"","make_diff_sim_upsilon"],[10,3,1,"","make_kernel_functionals"],[10,3,1,"","make_l1_metrics"],[10,3,1,"","make_l1_norm"],[10,3,1,"","make_l2_dot"],[10,3,1,"","make_l2_metrics"],[10,3,1,"","make_l2_norm"],[10,3,1,"","make_maps"],[10,3,1,"","make_offset_gen"],[10,3,1,"","make_phi"],[10,3,1,"","make_psi"],[10,3,1,"","make_upsilon"],[10,3,1,"","test_nullity"]],"aslsd.models":[[11,0,0,"-","exp_hawkes"],[11,0,0,"-","mhp"]],"aslsd.models.exp_hawkes":[[11,1,1,"","RecurrentExponential"],[11,4,1,"","diff_exp_lse_omega_kpr"],[11,4,1,"","diff_exp_lse_theta_kpr"],[11,4,1,"","get_exp_lse_k"],[11,4,1,"","get_exp_sum_reccurence_lag_h"],[11,4,1,"","update_exp_sum_recurrence"]],"aslsd.models.exp_hawkes.RecurrentExponential":[[11,3,1,"","diff_lse_mu_k"],[11,3,1,"","lse_k_grad"]],"aslsd.models.mhp":[[11,1,1,"","MHP"]],"aslsd.models.mhp.MHP":[[11,3,1,"","clear_fit"],[11,2,1,"","d"],[11,2,1,"","diff_cross_upsilon"],[11,2,1,"","diff_phi"],[11,2,1,"","diff_psi"],[11,2,1,"","diff_sim_upsilon"],[11,3,1,"","fit"],[11,2,1,"","fit_log"],[11,2,1,"","fit_residuals"],[11,2,1,"","fitted_adjacency"],[11,2,1,"","fitted_ker_param"],[11,2,1,"","fitted_mu"],[11,3,1,"","get_branching_ratio"],[11,3,1,"","get_l2_projection"],[11,3,1,"","get_n_param"],[11,3,1,"","get_param_bounds"],[11,3,1,"","get_param_names"],[11,3,1,"","get_random_param"],[11,3,1,"","get_residuals"],[11,2,1,"","interval_map"],[11,2,1,"","is_fitted"],[11,3,1,"","is_sbf"],[11,2,1,"","ix_map"],[11,2,1,"","ker_param_names"],[11,5,1,"id0","kernel_matrix"],[11,3,1,"","ks_test_residuals"],[11,3,1,"","load"],[11,3,1,"","make_adjacency_matrix"],[11,3,1,"","make_kernel_functionals"],[11,3,1,"","make_maps"],[11,3,1,"","matrix2tensor_params"],[11,2,1,"","matrix_n_param"],[11,2,1,"","mu_names"],[11,2,1,"","n_ker_param"],[11,2,1,"","param_bounds"],[11,2,1,"","phi"],[11,3,1,"","plot_adjacency_matrix"],[11,3,1,"","plot_kernels"],[11,3,1,"","plot_solver_path"],[11,2,1,"","psi"],[11,3,1,"","qq_plot"],[11,3,1,"","save"],[11,3,1,"","simu_multipath"],[11,3,1,"","simulate"],[11,3,1,"","tensor2matrix_params"],[11,3,1,"","tensor2matrix_solverpaths"],[11,2,1,"","upsilon"]],"aslsd.optim_logging":[[12,0,0,"-","optim_logger"]],"aslsd.optim_logging.optim_logger":[[12,1,1,"","OptimLogger"]],"aslsd.optim_logging.optim_logger.OptimLogger":[[12,2,1,"","d"],[12,2,1,"","estimator_logs"],[12,2,1,"","grad_ker"],[12,2,1,"","grad_logs"],[12,2,1,"","grad_mu"],[12,2,1,"","is_log_grad"],[12,2,1,"","is_log_param"],[12,2,1,"","ker"],[12,2,1,"","ker_0"],[12,3,1,"","log_grad"],[12,3,1,"","log_param"],[12,2,1,"","mu"],[12,2,1,"","mu_0"],[12,2,1,"","n_iter"],[12,2,1,"","param_logs"],[12,3,1,"","process_logs"]],"aslsd.plots":[[13,0,0,"-","graphic_tools"]],"aslsd.plots.graphic_tools":[[13,4,1,"","animate_plot_sequence"],[13,4,1,"","annotate_heatmap"],[13,4,1,"","get_horizontal_range_brace"],[13,4,1,"","get_vertical_range_brace"],[13,4,1,"","make_heatmap"],[13,4,1,"","plot_adjacency_matrix"],[13,4,1,"","plot_function_shaded_error"],[13,4,1,"","plot_kernels"],[13,4,1,"","plot_solver_path"],[13,4,1,"","plot_solver_path_contour"]],"aslsd.solvers":[[14,0,0,"-","adam"],[14,0,0,"-","momentum"],[14,0,0,"-","solver"]],"aslsd.solvers.adam":[[14,1,1,"","ADAM"]],"aslsd.solvers.adam.ADAM":[[14,3,1,"","iterate"]],"aslsd.solvers.momentum":[[14,1,1,"","Momentum"]],"aslsd.solvers.momentum.Momentum":[[14,3,1,"","iterate"]],"aslsd.solvers.solver":[[14,1,1,"","Solver"]],"aslsd.solvers.solver.Solver":[[14,3,1,"","iterate"],[14,3,1,"","make_learning_rate"]],"aslsd.utilities":[[15,0,0,"-","useful_functions"],[15,0,0,"-","useful_statistics"]],"aslsd.utilities.useful_functions":[[15,4,1,"","concatenate_linspace"],[15,4,1,"","discretize_space"],[15,4,1,"","double_sum_exp_1D"],[15,4,1,"","double_sum_exp_1D_midpoint"],[15,4,1,"","finite_diff"],[15,4,1,"","gammaf"],[15,4,1,"","gaussian_int"],[15,4,1,"","gaussian_pdf"],[15,4,1,"","get_alphabet_range"],[15,4,1,"","linc_gammaf"],[15,4,1,"","normal_cdf"],[15,4,1,"","prod_ratio"],[15,4,1,"","share_ratio"],[15,4,1,"","std_gamma_cdf"],[15,4,1,"","tri"],[15,4,1,"","update_sum_exp_1D"]],"aslsd.utilities.useful_statistics":[[15,4,1,"","batch_mean"],[15,4,1,"","batch_std"],[15,4,1,"","estimate_mean"],[15,4,1,"","estimate_std_wor_unbiased"],[15,4,1,"","get_ks_alpha"],[15,4,1,"","random"],[15,4,1,"","sample_double_ix_stratified"],[15,4,1,"","sample_double_ix_stratum"],[15,4,1,"","sample_single_ix_stratified"],[15,4,1,"","sample_single_ix_stratum"],[15,4,1,"","stratified_double_ixs2phi_times"],[15,4,1,"","stratified_double_ixs2upsilon_times"],[15,4,1,"","stratified_single_ixs2times"]],aslsd:[[1,0,0,"-","applications"],[2,0,0,"-","basis_kernels"],[4,0,0,"-","estimators"],[6,0,0,"-","evaluation"],[8,0,0,"-","events"],[10,0,0,"-","kernels"],[11,0,0,"-","models"],[12,0,0,"-","optim_logging"],[13,0,0,"-","plots"],[14,0,0,"-","solvers"],[15,0,0,"-","utilities"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function","5":"py:property"},terms:{"0":[2,8,10,11,13,15],"001":15,"07":9,"08":[9,13],"1":[2,6,8,10,11,13,14,15,18],"10":[11,13,15],"100":13,"1000":[6,10,11,13],"10637":18,"11":13,"12":14,"1234":[6,10,11,15],"145":14,"15":9,"151":14,"18":13,"19":18,"194566":13,"1987":[],"1999":14,"2":[2,11,14,15,18],"20":13,"200":[],"2012":13,"2015":14,"2021":[2,9,18],"2111":18,"22":2,"28":9,"2d":13,"2du":2,"2f":13,"3":[11,15,18],"30":[],"300":13,"3969208":[],"4":18,"5":[11,13],"50":13,"59":18,"6":[11,18],"7":18,"75":13,"\u00e1":2,"\u03bc":13,"\u03c9":13,"abstract":[2,4,14],"case":[2,11,13],"class":[2,4,8,10,11,12,13,14,17,18],"default":[2,11,13,15],"do":2,"final":11,"float":[8,11,13,15],"function":[2,8,11,13,15,17,18],"int":[2,8,11,12,13,15],"new":[11,13],"null":2,"public":18,"return":[2,8,11,15],"true":[2,4,6,10,11,13],"var":2,A:[2,8,10,11,13,14,18],By:[2,8,11,15],For:[10,11,13,15],If:[8,10,11,13,15,18],In:[2,11,14,15,18],It:15,On:14,That:2,The:[2,8,10,11,13,14,15],To:2,_1:2,_2:2,_:[2,8,10,11,14,15],_basis_kernel:10,_kernel_matrix:11,a_:[11,14],ab:[],abc:[2,4,14],abov:13,absolut:11,abus:2,accord:13,acknowledg:18,across:8,activ:18,adam:0,adapt:18,adaptive_stratified_estim:0,adaptivestratifi:4,add:2,addit:11,adjac:[11,13],adjacency_matrix:13,after:15,algorithm:[11,14,18],all:[2,8,10,11,13,15],alloc_typ:[],allow:2,allow_simu:[2,10],alpha:[2,15],alphabet:15,also:[],altitud:[2,15],an:[2,11,13,15],ani:2,anim:13,animate_plot_sequ:13,annot:[11,13],annotate_heatmap:13,api:16,append:10,appli:13,applic:[0,17,18],approxim:15,aquamarin:13,ar:[2,11,13,14,18],arbitrari:2,argument:[11,13],arrai:[2,8,10,11,13,15],arriv:[11,15],articl:18,arxiv:18,aslsd:17,associ:[2,8],assum:2,atrribut:11,attribut:[2,11],author:[9,18],auto:11,automat:11,avail:2,avoid:[2,14],ax:[6,11,13],axesimag:13,axi:13,axis_resolut:13,b:[2,10],b_1:2,b_2:[],b_:2,b_i:2,b_r:[],ba:14,background:[11,13],backward:15,bar:11,base:[2,4,8,10,11,12,13,14,18],base_se:11,baselin:11,basi:[2,10],basis_kern_2:2,basis_kernel:[0,10,17],basis_kernel_delayedexp:0,basis_kernel_exponenti:0,basis_kernel_gaussian:0,basis_kernel_nul:0,basis_kernel_rayleigh:0,basis_kernel_triangl:0,basisfunc_index:[],basiskernel:[2,10],batch_mean:15,batch_std:15,been:11,befor:15,belong:13,below:13,beta1:13,beta2:13,beta:[2,11,15],between:[2,11,15],bibtex:[],big:[2,10,15],bigg:[2,11,15],black:13,blog:13,blue:[11,13],boldsymbol:[8,11,14],book:[],bool:[2,11,12,13],bound:[2,11,15],brace:13,branch:11,broadcast:10,c:15,call:[11,13,18],can:[2,13],cartea2021gradi:18,cartea:[2,18],cbar_kw:13,cbarlabel:13,cdf:15,cdot:[2,14],central:15,character:2,check:2,chosen:15,chronolog:8,citat:16,clear:[11,15],clear_fit:11,close:[2,11,13],cluster:18,cmap:[11,13],code:[2,18],cohen:[2,18],col_label:13,color:13,colorbar:13,colormap:13,column:13,com:[],compar:18,compat:2,compens:11,complex:18,compon:[2,13],comput:[2,8,11,15],comupt:2,concaten:[2,10,11],concatenate_linspac:15,condit:[11,18],confer:14,confus:2,consid:15,constant:[],construct:13,contain:11,content:[],context:13,contour:13,contour_color:13,control:2,convent:8,convert:11,coordin:15,corner:[2,15],correl:11,correspond:[2,11,15],count:[8,11],creat:[9,13],cross:11,cumul:15,current:[13,15],d:[2,6,8,11,12,13],d_2:2,d_:2,darkorang:13,data:[11,13],dataset:18,decai:[2,15],deepskyblu:13,defin:[2,8,10,11,15],definit:2,delai:2,delayedexponentialkernel:2,delet:11,delta:[2,14,15],denot:[2,10,11,14],densiti:15,depend:[11,14,15,16],deriv:[2,11,13,15],derivatives_zero:[11,13],descent:[14,18],descript:[2,8,11,12,16],develop:18,deviat:[2,15],dict:[2,11],dict_interact:2,dictionari:[11,13],diederik:14,diff_cross_upsilon:[2,10,11],diff_cross_upsilon_delayedexp_delayedexp:2,diff_cross_upsilon_delayedexp_nul:2,diff_cross_upsilon_exp_exp:2,diff_cross_upsilon_exp_gauss:2,diff_cross_upsilon_exp_nul:2,diff_cross_upsilon_gauss_exp:2,diff_cross_upsilon_gauss_gauss:2,diff_cross_upsilon_gauss_nul:2,diff_cross_upsilon_rev:2,diff_exp_lse_omega_kpr:11,diff_exp_lse_theta_kpr:11,diff_index:15,diff_kl_diverg:[2,10],diff_kl_divergence_gauss_gauss:2,diff_l1_norm:[2,10],diff_l2_dot_delayedexp_delayedexp:2,diff_l2_dot_delayedexp_nul:2,diff_l2_dot_exp_exp:2,diff_l2_dot_exp_gauss:2,diff_l2_dot_exp_nul:2,diff_l2_dot_gauss_exp:2,diff_l2_dot_gauss_gauss:2,diff_l2_dot_gauss_nul:2,diff_l2_norm:2,diff_l2_projection_loss:10,diff_lse_mu_k:[4,11],diff_lse_theta_kpr:4,diff_phi:[2,10,11],diff_psi:[2,10,11],diff_sim_upsilon:[2,10,11],diff_typ:15,differ:[2,15],differenti:15,difficulti:18,dimens:[8,11,12],dimension:[8,11,13],directli:18,disc_typ:[11,15],discard:15,discret:15,discretize_spac:15,discuss:[2,13],displai:13,display_derivatives_zero:[],display_line45:[6,11],distanc:[2,15],distribut:[11,15],diverg:2,divid:[],divis:14,doc:[],document:17,doe:18,domain:2,dot:[2,15],doubl:15,double_ix_stratif:0,double_sum_exp_1d:15,double_sum_exp_1d_midpoint:15,doubleixstratif:4,dpi:13,dt:15,du:[2,11,15],e741:15,e:[2,13,14],e_m:11,e_n:[],each:[2,10,11,13],edward:[],eigenvalu:11,either:13,element:15,elementari:[],empti:2,end:15,entri:11,epsilon:15,equal:11,error:[11,18],esimt:11,estim:[0,2,11,13,15,17,18],estimate_dphi:4,estimate_dphi_adaptivedom:4,estimate_dphi_adaptivedom_gtarget:4,estimate_dphi_gtarget:4,estimate_dpsi:4,estimate_dupsilon:4,estimate_dupsilon_adaptivedom:4,estimate_dupsilon_adaptivedom_gtarget:4,estimate_dupsilon_gtarget:4,estimate_dupsilonzero:4,estimate_mean:15,estimate_std_wor_unbias:15,estimator_log:12,eta:[8,11],eta_:8,evalu:[0,11,15,17,18],evenli:15,event:[0,10,11,17,18],event_nam:[11,13],exact:18,exact_estim:0,exactestim:4,exactli:13,exampl:13,excit:18,exclud:15,exist:[8,18],exp:[2,15],exp_hawk:0,explicitli:2,exponenti:[2,6,11,13,15],exponential_mhp:[],exponentialkernel:2,expontenti:15,express:2,extrm:[],f:[2,10,15],f_:[2,15],fals:[2,4,6,10,11,13,15],famili:2,fast:18,favour:18,figsiz:[],figur:13,file:11,filenam:[6,11,13],finit:15,finite_diff:15,firebrick:13,first:[11,13,15],fit:[2,11,13,18],fit_log:[11,13],fit_residu:11,fitted_adjac:11,fitted_ker_param:11,fitted_mu:11,fix:2,fixed_indic:2,fixed_var:2,flat:11,flatten:11,follow:15,foot:[2,15],foral:11,form:[2,13,14],formal:15,format:13,formatt:13,formula:15,forward:[13,15],frac:[2,8,11,14,15],from:[2,11,13],full:17,func:[4,15],func_rate_decay_exp:[],func_rate_decay_lin:[],fundament:18,g:[2,13,14],g_:[2,14],g_t:12,galleri:13,gamma:15,gammaf:15,gaussian:[2,15],gaussian_int:15,gaussian_pdf:15,gaussiankernel:2,gener:[2,11,13,15,18],geq:[2,8,10,11,15],get:[2,11,15],get_alphabet_rang:15,get_branching_ratio:11,get_default_abs_alloc:4,get_default_adaptive_rel_alloc_0:4,get_default_adaptive_strata:4,get_default_nonadaptive_strata:4,get_default_strata:4,get_exp_lse_k:11,get_exp_sum_reccurence_lag_h:11,get_horizontal_range_brac:13,get_index_in_basi:[],get_interact:2,get_kappa_varpi:8,get_ks_alpha:15,get_l2_project:[10,11],get_l2_projection_loss:10,get_lag_s:8,get_lag_strata_finitepop:4,get_lag_strata_s:4,get_log:4,get_n_param:[2,11],get_n_var:2,get_param_bound:[2,10,11],get_param_nam:[2,11],get_random_param:11,get_residu:[6,11],get_residuals_hpoisson:6,get_residuals_k:6,get_residuals_k_hpoisson:6,get_stratification_s:4,get_true_allocation_phi:4,get_true_allocation_upsilon:4,get_var_bound:2,get_var_nam:2,get_vec_param_nam:10,get_vertical_range_brac:13,getting_start:[],github:13,given:[2,11,15],global:8,good:18,goodness_of_fit:0,grad:[4,14],grad_ker:12,grad_log:12,grad_mu:12,grad_upd:13,gradient:[2,11,14,18],graphic_tool:0,ground:13,guess:11,h:[8,11],h_inf:15,h_max:[4,9],h_sup:15,ha:[2,11,15],half:2,hand:11,has_omega_param:2,have:[2,11,15,18],hawk:[2,11,18],heatmap:13,height:13,higher:15,hom_poisson:[],homoegen:[],homogen:11,homogeneous_poisson:[],hompoisson:[],horizont:13,html:13,http:13,hyper:14,i:[2,4,8,11,15],i_m:8,i_t:8,igg:[],ight:[],ij:11,ijk:11,im:13,imag:[6,11,13],image_annotated_heatmap:13,images_contours_and_field:13,implement:[2,11,15,18],imshow:13,incomplet:15,independ:11,index:[8,11,15,16],index_from_on:[11,13],index_max:[],index_min:[],indexerror:8,indic:[2,11],induct:15,inform:11,infti:[2,8,11,15],initi:[4,11],initial_divis:13,inner:2,insid:13,instal:[],instanc:[2,13],instead:[],int_0:11,int_:[2,11,15],integ:[11,15],integr:15,intens:[11,18],inter:[11,15],interc:2,intern:14,interv:[2,13,15],interval_map:[10,11],intialize_log:4,io:13,is_compat:[2,10],is_fit:11,is_log_grad:12,is_log_param:12,is_nul:10,is_revers:2,is_sbf:[10,11],iter:[11,14],its:[2,11],ix_diff:2,ix_func:2,ix_map:[2,10,11],ix_omega:10,j:[4,8,11,15],jakevdp:13,jan:9,jimmi:14,journal:18,jump:11,k:[6,11,12,14,15],k_:[],k_t:[],kappa:[8,11,15],kei:11,ker:[11,12],ker_0:12,ker_param_2:10,ker_param_nam:[11,13],kernel:[0,2,11,13,17,18],kernel_2:10,kernel_matrix:11,kernel_param:[6,11,13],kernelmodel:[10,11],keyword:11,ki:11,kingma:14,kj:11,kl:2,kl_diverg:[2,10],kl_divergence_gauss_gauss:2,known:13,ks_test_residu:[6,11],kullback:2,kwarg:[4,6,10,11,12,13,14],l1_norm:[2,10],l2_distanc:[2,10],l2_dot:[2,10],l2_dot_delayedexp_delayedexp:2,l2_dot_delayedexp_nul:2,l2_dot_exp_exp:2,l2_dot_exp_gauss:2,l2_dot_exp_nul:2,l2_dot_gauss_exp:2,l2_dot_gauss_gauss:2,l2_dot_gauss_nul:2,l2_norm:[2,10],l:15,l_:[2,11],label:[6,11,13],labyad:[2,18],lag_siz:[4,8],lambda:11,lambda_:11,lambda_k:11,langl:2,larg:18,last:11,latest:8,ldot:11,learn:14,learning_r:[],least:[8,11,18],leav:2,left:[2,8,11,14,15],leibler:2,length:13,leq:[2,15],let:[2,10,11,15],letter:15,like:2,likelihood:18,limegreen:13,lin:13,linc_gammaf:15,line:13,linear:[2,11,15,18],list:[2,8,10,11,12,13,15],list_color:13,list_i:13,list_label:13,list_r:[],list_sampl:15,list_tim:[6,8,9,11,15],list_times2end:[8,11,15],list_y_ref:13,list_y_std:13,list_y_v:13,live:2,load:11,log:[2,11,15,18],log_error:[10,11],log_grad:12,log_param:12,log_scal:[6,11,13],logarithm:15,logger:[11,12],loglikelihood:[],loss_funct:13,lower:[2,11,15],lse:[11,13,18],lse_k_estim:4,lse_k_grad:11,lse_k_grad_estim:4,lvaro:18,m:[8,11,13,14,15],m_:11,m_inf:15,m_max:9,m_sup:15,magma:13,main:18,make:13,make_adjacency_matrix:11,make_dict_interact:2,make_diff_cross_upsilon:[2,10],make_diff_cross_upsilon_rev:2,make_diff_kl_diverg:2,make_diff_kl_divergence_rev:2,make_diff_l1_norm:[2,10],make_diff_l2_dot:[2,10],make_diff_l2_norm:[2,10],make_diff_phi:[2,10],make_diff_psi:[2,10],make_diff_sim_upsilon:[2,10],make_heatmap:13,make_ix_map:2,make_kernel_funct:[2,10,11],make_kl_diverg:2,make_kl_divergence_rev:2,make_kl_funct:2,make_l1_metr:[2,10],make_l1_norm:[2,10],make_l2_dot:[2,10],make_l2_metr:[2,10],make_l2_norm:[2,10],make_learning_r:14,make_map:[10,11],make_offset_gen:10,make_phi:[2,10],make_psi:[2,10],make_semi_uniform_partit:[],make_simu_func:2,make_uniform_partit:[],make_upsilon:[2,10],make_upsilon_rev:2,make_var:2,map:[11,13],mat_n_param:11,math:[],mathbb:[2,8,11,15],mathbbm:[],mathbf:11,mathcal:[8,11,14,15],mathd:[],mathrm:[2,11,14],matplotlib:13,matrix2tensor_param:11,matrix:[11,13],matrix_kappa:9,matrix_lag_s:8,matrix_n_param:[11,13],matrix_varpi:9,max:15,max_mu:11,max_omega:11,max_param:11,max_point:[6,11],maxim:[],maximum:11,mean:[2,15],mean_1:15,mean_2:15,mediumorchid:13,method:[2,13,14,17,18],metric:2,mhp:[0,2,10,12,13,18],mhp_2:11,mhp_model:4,mhp_simul:[],mid:13,middl:13,midpoint:15,min:15,min_i:13,min_ker_param:[11,13],min_mu:[11,13],min_param_color:13,min_x:13,minim:13,model:[0,8,10,13,17],modifi:11,modul:[16,17],momentum:0,move:15,mu:[6,11,12],mu_0:12,mu_:11,mu_k:[4,6,11],mu_nam:[11,13],multivari:[11,18],must:11,n:[2,8,11,13,15,18],n_:[2,8,10,11],n_b:[2,10],n_basis_k:10,n_bucket:[],n_event:[8,11,15],n_events_i:4,n_fixed_var:2,n_iter:[10,11,12],n_k:11,n_ker_param:11,n_model:6,n_param:10,n_r:15,n_sampl:[11,13,15],name:[2,11,13],ndarrai:[2,8,10,11,12,15],need:8,nelson:[],net:13,network:14,neural:14,newer:18,ning:14,non:2,none:[2,4,6,8,9,10,11,13,15],nonzero:2,norm:[2,11],normal:[6,11,15],normal_cdf:15,note:[2,13,14,15],notimplementederror:2,notion:2,novemb:2,now:11,np:8,nullkernel:2,number:[2,8,11,15,18],numpi:[2,8,10,11,12,13,15,18],o:15,obj_2:10,object:[2,4,8,10,11,12],observ:[11,18],obtainend:[2,10],odot:14,oldsymbol:[],omega:2,one:[2,13],onli:[2,11,13],open:2,oper:14,optim:[2,11,12,14,18],optim_log:[0,17],optim_logg:0,optimis:11,optimlogg:[11,12],option:[2,11,13,15],order:8,orderli:8,org:13,other:[2,11,13],otherwis:2,our:[2,11,15,18],output:2,own:15,p:[4,11,14],packag:[0,15,17],pad:[],page:16,panda:18,paper:18,par:11,param:[2,10,11],param_2:11,param_bound:11,param_log:12,param_nam:[],param_upd:13,paramet:[2,8,10,11,13,14,15,18],parametr:[2,10,13,18],params_0:10,paramt:2,part:[15,18],partial:2,particular:2,path:[8,11,13],path_color:13,path_r:11,pdf:15,per:11,perform:[14,15],period:[],phi:[2,10,11,13],phi_:[2,10,11],pi:[2,15],pleas:18,plot:[0,17],plot_adjacency_matrix:[11,13],plot_deriv:[11,13],plot_function_shaded_error:13,plot_kernel:[11,13],plot_solver_path:[11,13],plot_solver_path_contour:13,png:[6,11,13],point:[8,11,18],poisson:11,posit:[2,15],potenti:15,practic:18,preced:8,predecessor:8,preprint:18,press:[],previou:15,previous:11,prime:[2,15],primit:11,princeton:[],print:11,probabl:15,problem:11,process:[2,8,11,18],process_log:12,process_path:[0,4,6,11],processpath:[8,11],prod_ratio:15,product:2,progress:11,prohibit:18,proj:14,project:14,properti:[10,11],proport:2,propos:18,provid:13,psi:[2,6,10,11],psi_:11,publish:[],py4u:13,pycodestyl:15,pydata:[],python:18,q:[11,15],q_1:15,q_2:15,q_alloc:[],qeg:[],qian:14,qq_plot:[6,11],quad:8,quadrat:18,quantiti:15,r:[2,4,10,11,15],radic:[],radiu:11,rais:[2,8,11,15],random:[8,11,15],range_ref:11,rangle_2:2,rate:[2,8,11,13,14,15],ratio:11,rayleigh:2,rayleighkernel:2,re:15,real:15,recurrentexponenti:11,ref_ker_param:11,ref_mu:11,ref_x:13,refer:[2,16],remain:8,replac:15,repres:11,represent:14,requir:18,residu:[6,11],resolution_factor:13,resp:11,respect:13,result:[11,18],rho_:11,right:[2,8,11,14,15],rng:[2,4,10,11,15],roman:15,row:13,row_label:13,s:[2,11,13,15],s_:[11,15],s_m:15,s_n:15,saad:18,saadl:9,sai:11,same:13,sampl:[6,11,15],sample_double_ix_stratifi:15,sample_double_ix_stratum:15,sample_s:[6,11],sample_single_ix_stratifi:15,sample_single_ix_stratum:15,samuel:18,satisfi:2,save:[6,11,13],scale:[2,13,15],scientif:18,scipi:[15,18],search:16,second:13,see:2,seed:[4,6,10,11,15],self:18,sens:13,separ:13,sequenc:[8,13,15],set:[2,11,14,15],set_doublesum_estim:4,set_singlesum_estim:4,set_stratif:4,sgd:14,shape:[13,15],share_ratio:15,should:13,show:[6,11,13],side:11,sim_func:[2,10],simplifi:11,simu_multipath:11,simul:[11,13,18],sinc:11,single_ix_stratif:0,singleixstratif:4,size:[2,11],small:14,smaller:8,smallest:8,solver:[0,10,11,13,17],some:[2,8,11],sort:8,space:[13,15],specif:13,specifi:[2,11],spectral:11,sqrt:[2,14,15],squar:[11,14,18],ssrn:2,stabl:13,standard:[2,11,15],start:[11,15],std:15,std_1:15,std_2:15,std_gamma_cdf:15,std_offset:[],steelblu:13,step:15,stochast:[14,18],store:11,str:[11,13,15],strata:[4,15],strata_s:[],stratifi:18,stratified_double_ixs2phi_tim:15,stratified_double_ixs2upsilon_tim:15,stratified_single_ixs2tim:15,string:13,structur:2,style:[6,11,15],submodul:0,subpackag:17,subsampl:11,substract_yx:[6,11],sum:[2,10,15],sum_:[2,8,10,11,15],suppos:11,t:[2,8,10,11,12,14,15],t_0:15,t_1:15,t_2:15,t_:[8,11,15],t_f:[8,11,15],t_k:15,t_m:[8,15],t_max:[11,13],t_min:[11,13],t_n:15,t_re:11,taken:15,target_bratio:11,tensor2matrix_param:11,tensor2matrix_solverpath:11,tensor:[11,13],tensor_param:11,tensor_path:11,term:[11,14],termin:[8,11,15],test:[0,2,4,6,8,18],test_kappa:9,test_nul:10,test_varkappa:9,text:[8,11,13],textcolor:13,textkw:13,textrm:[2,10,15],th:11,than:[8,15],thei:11,them:8,theori:[],therefor:11,theta:[2,10,11],theta_:[11,14],thi:[2,8,11,13,14,15,18],those:13,threshold:13,thu:9,ticker:13,tild:2,time:[8,11,14,15],time_ord:0,time_ordering_test:[0,8],titl:18,top:11,total:11,tqdm:18,tree:[],tri:15,triangl:[2,15],triangular:2,triangularkernel:2,tright:[],true_i:13,true_ker_param:[11,13],true_mu:[11,13],true_omega:11,true_param_color:13,true_x:13,truth:13,tutori:13,two:[11,13,15,18],type:[2,8,11,12,15],u:[2,11,15],unbias:15,under:18,unidimension:10,uniform:[],unit:13,univers:[],up:[8,11],updat:[13,14,15],update_exp_sum_recurr:11,update_sum_exp_1d:15,upper:15,upsilon:[2,10,11],upsilon_:11,upsilon_delayedexp_delayedexp:2,upsilon_delayedexp_nul:2,upsilon_exponential_exponenti:2,upsilon_exponential_gaussian:2,upsilon_exponential_nul:2,upsilon_gauss_exp:2,upsilon_gauss_gauss:2,upsilon_gauss_nul:2,upsilon_rev:2,us:[2,11,13,15,18],useful_funct:0,useful_statist:0,user:2,util:[0,17],valfmt:13,valu:[2,11,13,15],valueerror:[11,15],variabl:[2,15],variablei:[],varianc:15,varphi:2,varpi:[8,11,15],vars_1:2,vars_2:2,vars_:2,vartheta:[2,10],vartheta_1:2,vartheta_:[2,11],vartheta_i:2,vartheta_p:2,vartheta_r:[],vec_n_param:10,vec_n_r:15,vec_t:15,vector:[2,10,11,13,15],verbos:[6,10,11],veri:14,version:15,vertic:13,video:13,we:[2,11,15,18],weight:[],well:[11,18],wether:2,when:[2,13,18],where:[2,8,11,13],which:[11,13,15,18],white:13,whitesmok:13,write:11,www:13,x:[2,13,14,15],x_0:11,x_:15,x_k:[4,11,12],x_ki:4,x_kj:4,x_max:[11,13,15],x_min:[11,13,15],x_name:13,x_updat:13,x_val:13,xname:13,xy:15,y:[13,15],y_max:13,y_min:13,y_name:13,y_updat:13,year:18,yname:13,you:18,z_:15,z_n:15,zero:14},titles:["aslsd","aslsd.applications","aslsd.basis_kernels","aslsd.basis_kernels.tests","aslsd.estimators","aslsd.estimators.tests","aslsd.evaluation","aslsd.evaluation.tests","aslsd.events package","aslsd.events.tests","aslsd.kernels","aslsd.models","aslsd.optim_logging","aslsd.plots","aslsd.solvers","aslsd.utilities","Welcome to aslsd\u2019s documentation!","API reference","aslsd"],titleterms:{adam:14,adaptive_stratified_estim:4,api:17,applic:1,aslsd:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18],basis_kernel:[2,3],basis_kernel_delayedexp:2,basis_kernel_exponenti:2,basis_kernel_gaussian:2,basis_kernel_nul:2,basis_kernel_rayleigh:2,basis_kernel_triangl:2,citat:18,content:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],depend:18,descript:18,document:16,double_ix_stratif:4,estim:[4,5],evalu:[6,7],event:[8,9],exact_estim:4,exp_hawk:11,exponential_mhp:[],goodness_of_fit:6,graphic_tool:13,hom_poisson:[],homogeneous_poisson:[],indic:16,kernel:10,mhp:11,mhp_simul:[],model:11,modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],momentum:14,optim_log:12,optim_logg:12,packag:8,plot:13,process_path:8,refer:17,s:16,simul:[],single_ix_stratif:4,solver:14,submodul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],subpackag:[0,2,4,6,8],tabl:[16,17],test:[3,5,7,9],time_ord:8,time_ordering_test:9,tree:[],useful_funct:15,useful_statist:15,util:15,welcom:16}})