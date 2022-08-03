.PHONY: mnist-adapt clean-mnist-adapt mnist-gmm clean-mnist-gmm mnist-embedder clean-mnist-embedder clean-mnist


mnist-adapt:
	JAX_PLATFORM_NAME=cpu parallel \
		--eta \
		--header : \
		--joblog mnist/adapt.joblog \
		pipenv run python3 \
		-m mnist.adapt_gmm \
		--embedder_name {embedder_name} \
		--embedder_dim {embedder_dim} \
		--embedder_lr {embedder_lr} \
		--embedder_epochs {embedder_epochs} \
		--gmm_init {gmm_init} \
		--gmm_k {gmm_k} \
		--gmm_r {gmm_r} \
		--gmm_lr {gmm_lr} \
		--gmm_dis {gmm_dis} \
		--gmm_un {gmm_un} \
		--gmm_epochs {gmm_epochs} \
		--adapt_deg {adapt_deg} \
		--adapt_lr {adapt_lr} \
		--adapt_epochs {adapt_epochs} \
		:::: grid/embedder_name \
		:::: grid/embedder_dim \
		:::: grid/embedder_lr \
		:::: grid/embedder_epochs \
		:::: grid/gmm_init \
		:::: grid/gmm_k \
		:::: grid/gmm_r \
		:::: grid/gmm_lr \
		:::: grid/gmm_dis \
		:::: grid/gmm_un \
		:::: grid/gmm_epochs \
		:::: grid/adapt_deg \
		:::: grid/adapt_lr \
		:::: grid/adapt_epochs


clean-mnist-adapt:
	for dir in logs ckpts; do \
		rm -f mnist/$$dir/adapt/*; \
	done


mnist-gmm:
	JAX_PLATFORM_NAME=cpu parallel \
		--eta \
		--header : \
		--joblog mnist/gmm.joblog \
		pipenv run python3 \
		-m mnist.train_gmm \
		--embedder_name {embedder_name} \
		--embedder_dim {embedder_dim} \
		--embedder_lr {embedder_lr} \
		--embedder_epochs {embedder_epochs} \
		--gmm_init {gmm_init} \
		--gmm_k {gmm_k} \
		--gmm_r {gmm_r} \
		--gmm_lr {gmm_lr} \
		--gmm_dis {gmm_dis} \
		--gmm_un {gmm_un} \
		--gmm_epochs {gmm_epochs} \
		:::: grid/embedder_name \
		:::: grid/embedder_dim \
		:::: grid/embedder_lr \
		:::: grid/embedder_epochs \
		:::: grid/gmm_init \
		:::: grid/gmm_k \
		:::: grid/gmm_r \
		:::: grid/gmm_lr \
		:::: grid/gmm_dis \
		:::: grid/gmm_un \
		:::: grid/gmm_epochs


clean-mnist-gmm:
	for dir in logs ckpts; do \
		rm -f mnist/$$dir/gmm/*; \
	done


mnist-embedder:
	JAX_PLATFORM_NAME=cpu parallel \
		--eta \
		--header : \
		--joblog mnist/embedder.joblog \
		pipenv run python3 \
		-m mnist.train_embedder \
		--embedder_name {embedder_name} \
		--embedder_dim {embedder_dim} \
		--embedder_lr {embedder_lr} \
		--embedder_epochs {embedder_epochs} \
		:::: grid/embedder_name \
		:::: grid/embedder_dim \
		:::: grid/embedder_lr \
		:::: grid/embedder_epochs


clean-mnist-embedder:
	for dir in logs ckpts; do \
		rm -f mnist/$$dir/embedder/*; \
	done


clean:
	clean-embedder
	clean-gmm
	clean-adapt
