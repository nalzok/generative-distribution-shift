.PHONY: mnist-gmm mnist-ae

mnist-gmm:
	parallel \
		--eta \
		--joblog mnist/gmm.joblog \
		pipenv run python3 \
		-m mnist.train_gmm \
		--embedding_dim {1} \
		--ae_lr {2} \
		--ae_ckpt_dir mnist/ckpts/ae \
		--unlabeled_factor {3} \
		--init_scheme {4} \
		--k {5} \
		--r {6} \
		--gmm_lr {7} \
		--lambda_ {8} \
		--kappa {9} \
		--epochs 8 \
		--gmm_ckpt_dir mnist/ckpts/gmm \
		'>' mnist/logs/gmm/gmm_dim{1}_aelr{2}_ufactor{3}_{4}_K{5}_R{6}_lr{7}_lambda{8}_kappa{9}.txt \
		:::: grid/dim \
		:::: grid/aelr \
		:::: grid/ufactor \
		:::: grid/init \
		:::: grid/K \
		:::: grid/R \
		:::: grid/gmmlr \
		:::: grid/lambda \
		:::: grid/kappa

mnist-ae:
	parallel \
		--eta \
		--joblog mnist/ae.joblog \
		pipenv run python3 \
		-m mnist.train_ae \
		--embedding_dim {1} \
		--ae_lr {2} \
		--epochs 32 \
		--ae_ckpt_dir mnist/ckpts/ae \
		'>' mnist/logs/ae/ae_dim{1}_lr{2}.txt \
		:::: grid/dim \
		:::: grid/aelr
