.PHONY: env download data dataset parse teacher teacher-all labels trainA trainB trainC eval vis pack
.PHONY: three-samples thousand-samples

env:
	pip install -r requirements.txt

data dataset:
	python scripts/download_dataset.py --dataset cubicasa5k

download: data
	python scripts/download_floorplancad.py
	# optional: python scripts/download_zind.py

parse:
	python scripts/parse_raster.py
	python scripts/parse_vector.py

teacher:
	python teacher/generate_labels.py --cfg configs/data.yaml --out data/labels/train.jsonl

teacher-all:
	python teacher/generate_labels.py --cfg configs/data.yaml --out data/labels/train.jsonl --limit 100000 --per_image 3

three-samples:
	python teacher/generate_labels.py --cfg configs/data.yaml --out data/labels/samples.jsonl --limit 3

thousand-samples:
	python teacher/generate_labels.py --cfg configs/data.yaml --out data/labels/train.jsonl --limit 1000

labels:
	python teacher/critic_filter.py --in data/labels/train.jsonl --out data/labels/train.filtered.jsonl

trainA:
	python student/train_stageA_answer_kd.py --cfg configs/trainA.yaml

trainB:
	python student/train_stageB_evidence_kd.py --cfg configs/trainB.yaml --ckpt outputs/*/student-A.pt

trainC:
	python student/train_stageC_dpo_unknown.py --cfg configs/trainC.yaml --ckpt outputs/*/student-B.pt

eval:
	python eval/evaluate.py --cfg configs/data.yaml --ckpt outputs/*/student-C.pt

vis:
	python eval/visualize_evidence.py --ckpt outputs/*/student-C.pt --out reports/vis/

pack:
	python scripts/package_release.py
