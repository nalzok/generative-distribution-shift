.PHONY: adapt clean-adapt gmm clean-gmm embeddings clean-embeddings clean


adapt:
	JAX_PLATFORM_NAME=cpu parallel \
		--eta \
		--header : \
		--joblog tta/adapt.joblog \
		pipenv run python3 \
		-m tta.adapt_gmm \
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


clean-adapt:
	for dir in logs ckpts; do \
		rm -f tta/$$dir/adapt/*; \
	done


gmm:
	JAX_PLATFORM_NAME=cpu parallel \
		--eta \
		--header : \
		--joblog tta/gmm.joblog \
		pipenv run python3 \
		-m tta.train_gmm \
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


clean-gmm:
	for dir in logs ckpts; do \
		rm -f tta/$$dir/gmm/*; \
	done


embeddings:
	parallel \
		--eta \
		--header : \
		--jobs 2 \
		--joblog embed/joblog \
		pipenv run python3 \
		-m embed.transformer \
		--split {embedding_split} \
		--model {embedding_model} \
		--global_pool {embedding_global_pool} \
		--mask_ratio {embedding_mask_ratio} \
		:::: grid/embedding_split \
		:::: grid/embedding_model \
		:::: grid/embedding_global_pool \
		:::: grid/embedding_mask_ratio


clean-embeddings:
	rm -rf data/embeddings


clean:
	clean-embeddings
	clean-gmm
	clean-adapt
