.PHONY: adapt clean-adapt gmm clean-gmm embeddings clean-embeddings clean


adapt:
	JAX_PLATFORM_NAME=cpu parallel \
		--eta \
		--header : \
		--joblog tta/adapt.joblog \
		pipenv run python3 \
		-m tta.adapt_gmm \
		--embedding_model {embedding_model} \
		--embedding_global_pool {embedding_global_pool} \
		--embedding_mask_ratio {embedding_mask_ratio} \
		--gmm_init {gmm_init} \
		--gmm_k {gmm_k} \
		--gmm_r {gmm_r} \
		--gmm_lr {gmm_lr} \
		--gmm_dis {gmm_dis} \
		--gmm_un {gmm_un} \
		--gmm_epochs {gmm_epochs} \
		--adapt_corruption {adapt_corruption} \
		--adapt_severity {adapt_severity} \
		--adapt_algo {adapt_algo} \
		--adapt_lr {adapt_lr} \
		--adapt_epochs {adapt_epochs} \
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
		:::: grid/adapt_corruption \
		:::: grid/adapt_severity \
		:::: grid/adapt_algo \
		:::: grid/adapt_lr \
		:::: grid/adapt_epochs


clean-adapt:
	rm -f logs/adapt/*
	rm -f ckpts/adapt/*


gmm:
	JAX_PLATFORM_NAME=cpu parallel \
		--eta \
		--header : \
		--joblog tta/gmm.joblog \
		pipenv run python3 \
		-m tta.train_gmm \
		--embedding_model {embedding_model} \
		--embedding_global_pool {embedding_global_pool} \
		--embedding_mask_ratio {embedding_mask_ratio} \
		--gmm_init {gmm_init} \
		--gmm_k {gmm_k} \
		--gmm_r {gmm_r} \
		--gmm_lr {gmm_lr} \
		--gmm_dis {gmm_dis} \
		--gmm_un {gmm_un} \
		--gmm_epochs {gmm_epochs} \
		:::: grid/embedding_model \
		:::: grid/embedding_global_pool \
		:::: grid/embedding_mask_ratio \
		:::: grid/gmm_init \
		:::: grid/gmm_k \
		:::: grid/gmm_r \
		:::: grid/gmm_lr \
		:::: grid/gmm_dis \
		:::: grid/gmm_un \
		:::: grid/gmm_epochs


clean-gmm:
	rm -f logs/gmm/*
	rm -f ckpts/gmm/*


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
	rm -rf data/embeddings/*


clean:
	clean-embeddings
	clean-gmm
	clean-adapt
