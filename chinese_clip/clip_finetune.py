#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @ Date: 2022-08-05 16:40
# @ Author: NING MEI


import os
import time
import json
import logging
from math import ceil
from pathlib import Path
from time import gmtime, strftime

import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from clip.clip import load
from clip.model import convert_weights, CLIP

from utils.data import get_data
from utils.params import parse_args
from utils.logger import setup_primary_logging, setup_worker_logging
from utils.scheduler import cosine_lr
from utils.save_best import *


def convert_models_to_fp32(model):
	for p in model.parameters():
		p.data = p.data.float()
		if p.grad:
			p.grad.data = p.grad.data.float()



def load_model(args):
	""" 加载模型 """
	print(" model loading ... ")
	path = Path(__file__).parent
	vision_model_config_file = os.path.join(path, f"model_configs/{args.vision_model}.json")
	text_model_config_file = os.path.join(path, f"model_configs/{args.text_model}.json")

	assert os.path.exists(vision_model_config_file) and os.path.exists(text_model_config_file), \
		f'{vision_model_config_file}, {text_model_config_file} are non-setting.'

	# 图文模型配置文件
	with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
		model_config = json.load(fv)
		for k, v in json.load(ft).items():
			model_config[k] = v

	# 图文权重文件
	if args.clip_weight_path is not None:
		assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
	if args.bert_weight_path is not None:
		assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"

	# CLIP模型加载
	model = CLIP(**model_config)
	load(model=model, device = args.device, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path)

	if args.precision == "amp" or args.precision == "fp32":
		convert_models_to_fp32(model)

	if args.device == 'cuda':
		print(args.device)
		# model.cuda()
	if args.freeze_vision:
		for k, v in model.visual.named_parameters():
			v.requires_grad = False
		logging.info("The visual encoder is freezed during training.")

	if args.precision == "fp16":
		convert_weights(model)

	return model



def get_loss(model, images, texts, loss_img, loss_txt, args):
	""" 计算文图余璇相似度 及 与类交叉熵 """
	image_features, text_features, logit_scale = model(images, texts)
	logit_scale = logit_scale.mean()

	# 余璇相似度logit
	logits_per_image = logit_scale * image_features @ text_features.t()
	logits_per_text = logit_scale * text_features @ image_features.t()

	# 交叉熵
	ground_truth = torch.arange(len(logits_per_image)).long()
	ground_truth = ground_truth.cuda(non_blocking=True) # 配合pin_memory=True 加快训练
	total_loss = ( loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth) ) / 2 ## 文图交叉熵平均值

	# acc = None
	# if args.report_training_batch_acc:
	# 	i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
	# 	t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
	# 	acc = {"i2t": i2t_acc, "t2i": t2i_acc}
	#
	i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
	t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
	acc = {"i2t": i2t_acc, "t2i": t2i_acc}
	return total_loss, acc



def evaluate(model, data, epoch, args, steps):
	""" 模型评估 """
	# logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))

	model.eval()

	dataloader = data['val'].dataloader
	data_iter = iter(dataloader)

	loss_img = nn.CrossEntropyLoss()
	loss_txt = nn.CrossEntropyLoss()

	loss_img = loss_img.cuda(args.local_device_rank)
	loss_txt = loss_txt.cuda(args.local_device_rank)

	cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
	cumulative_i2t_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
	cumulative_t2i_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
	num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
	all_image_features, all_text_features = [], []
	with torch.no_grad():
		for i in range(dataloader.num_batches):
			batch = next(data_iter)
			images, texts, eos_indices = batch

			images = images.cuda(args.local_device_rank, non_blocking=True)
			texts = texts.cuda(args.local_device_rank, non_blocking=True)
			eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

			image_features, text_features, logit_scale = model(images, texts)
			all_image_features.append(image_features)
			all_text_features.append(text_features)
			logit_scale = logit_scale.mean()
			logits_per_image = logit_scale * image_features @ text_features.t()
			logits_per_text = logits_per_image.t()

			ground_truth = torch.arange(len(images)).long()
			ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
			total_loss = ( loss_img(logits_per_image, ground_truth)  + loss_txt(logits_per_text, ground_truth)) / 2

			batch_size = len(images)
			cumulative_loss += total_loss * batch_size
			num_elements += batch_size

			cumulative_i2t_acc += ((logits_per_image.argmax(-1) == ground_truth).sum()).float()
			cumulative_t2i_acc += (logits_per_text.argmax(-1) == ground_truth).sum().float()

			if (i + 1) % 100 == 0:
				logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))


		loss = cumulative_loss / num_elements
		i2t_acc = cumulative_i2t_acc / num_elements
		t2i_acc = cumulative_t2i_acc / num_elements

		assert num_elements.item() == dataloader.num_samples # sanity check

		valid_loss = float(f"{loss.item():.6f}") # 累计平均损失
		image2text_acc = float(f"{i2t_acc.item() * 100:.2f}")
		text2image_acc = float(f"{t2i_acc.item() * 100:.2f}")
		logit_scale = float(f"{model.logit_scale.data:.3f}")

		valid_log_result = {"epoch": epoch + 1, "steps": steps, "loss": valid_loss, "image2text_acc": image2text_acc, \
		             "text2image_acc": text2image_acc, "logit_scale": logit_scale }

		VALID_LOG.append(valid_log_result)

		logging.info(
		    f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
		    f"Valid Loss: {valid_loss} | "
		    f"Image2Text Acc: {image2text_acc} | " 
		    f"Text2Image Acc: {text2image_acc} | " 
		    f"logit_scale: {logit_scale} | "
		    f"Valid Batch Size: {batch_size}"
		)

		return valid_log_result



TRAIN_LOG = []
VALID_LOG = []

def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps):

	model.train()

	dataloader, sampler = data['train'].dataloader, data['train'].sampler

	loss_img = nn.CrossEntropyLoss()
	loss_txt = nn.CrossEntropyLoss()

	loss_img = loss_img.cuda(args.local_device_rank)
	loss_txt = loss_txt.cuda(args.local_device_rank)

	if sampler is not None:
		sampler.set_epoch(epoch)

	num_batches_per_epoch = dataloader.num_batches
	data_iter = iter(dataloader)

	end = time.time()
	epoch_trained_steps = 0
	for i in range(global_trained_steps - num_batches_per_epoch * epoch, num_batches_per_epoch):
		batch = next(data_iter)
		step = num_batches_per_epoch * epoch + i
		# reach the args.max_steps, exit training:
		if step >= args.max_steps:
			logging.info("Stopping training due to step {} has reached max_steps {}".format(step, args.max_steps))
			return epoch_trained_steps
		scheduler(step)
		optimizer.zero_grad()
		images, texts, eos_indices = batch

		images = images.cuda(args.local_device_rank, non_blocking=True)
		texts = texts.cuda(args.local_device_rank, non_blocking=True)
		eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)  ## tokens

		data_time = time.time() - end

		m = model

		# with automatic mixed precision.
		if args.precision == "amp":
			with autocast():
				total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
				scaler.scale(total_loss).backward()
				scaler.step(optimizer)
			scaler.update()

		else:
			# 反向传播
			total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
			total_loss.backward()
			optimizer.step()

		# Note: we clamp to 4.6052 = ln(100), as in the original paper.
		m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
		batch_time = time.time() - end
		end = time.time()
		epoch_trained_steps += 1

		train_loss = float(f"{total_loss.item():.6f}")
		train_i2t_acc = float(f"{acc['i2t'].item() * 100:.2f}")
		train_t2i_acc = float(f"{acc['t2i'].item() * 100:.2f}")
		train_logit_scale = float(f"{m.logit_scale.data:.3f}")
		train_log_result = {"epoch": epoch + 1, "steps": step, "loss": train_loss, "image2text_acc": train_i2t_acc,
		                    "text2image_acc": train_t2i_acc, "logit_scale": train_logit_scale }
		TRAIN_LOG.append(train_log_result)

		if ((step + 1) % args.log_interval) == 0:
			num_samples = (i + 1) * len(images)
			samples_per_epoch = dataloader.num_samples
			percent_complete = 100.0 * (i + 1) / num_batches_per_epoch

			logging.info(
				f"Global Steps: {step + 1}/{args.max_steps} | " +
				f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
				f"Loss: {train_loss} | " +
				(f"Image2Text Acc: {train_i2t_acc} | " ) +
				(f"Text2Image Acc: {train_t2i_acc} | " ) +
				f"Data Time: {data_time:.3f}s | " +
				f"Batch Time: {batch_time:.3f}s | " +
				f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
				f"logit_scale: {train_logit_scale} | " +
				f"Global Batch Size: {len(images)}"
			)


		# epoch-steps 验证评估
		step_eval_res = {}
		num_patence = 3
		steps_early_stopping = EarlyStopping(patience=num_patence, verbose=args.debug)
		if args.val_data is not None and args.valid_step_interval is not None and ((step + 1) % args.valid_step_interval) == 0:
			assert "val" in data, "Error: Valid dataset has not been built."
			step_eval_res = evaluate(model, data, epoch, args, step + 1)

		if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
			save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_best.pt")

			## 模型评估
			if step_eval_res:
				step_loss = step_eval_res['loss']
				steps_early_stopping(step_loss, args, epoch + 1, step, model, optimizer, save_path)

			# Save the latest params
			save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_latest.pt")
			save_weights(args.name, epoch + 1, steps, model, optimizer, save_path)
			logging.info(
				"Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
				                                                                             step + 1,
				                                                                             time.time() - t1))

	return epoch_trained_steps, train_log_result



def main():
	time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
	args = parse_args()
	args.local_device_rank = 0
	args.rank = -1
	args.lr = 5e-5 # 设置lr
	print(args)

	device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device > -1 else "cpu")
	args.device = device
	args.log_path = os.path.join(args.logs, args.name, "out_{}.log".format(time_suffix))
	args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
	for dirname in [args.checkpoint_path]:
		if dirname:
			os.makedirs(dirname, exist_ok=True)

	# Set logger
	args.log_level = logging.DEBUG if args.debug else logging.INFO
	log_queue = setup_primary_logging(args.log_path, args.log_level, args.rank)
	setup_worker_logging(args.rank, log_queue, args.log_level)

	model = load_model(args)

	# Initialize dataset and dataloader
	data = get_data(args, epoch_id=0, max_txt_length=args.context_length)

	# Initialize optimizer and lr scheduler
	exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
	include = lambda n: not exclude(n)

	named_parameters = list(model.named_parameters())
	gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
	rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

	if args.train_data is None:
		optimizer = None
		scheduler = None
	else:
		optimizer = optim.AdamW(
			[
				{"params": gain_or_bias_params, "weight_decay": 0.},
				{"params": rest_params, "weight_decay": args.wd},
			],
			lr=args.lr,
			betas=(args.beta1, args.beta2),
			eps=args.eps,
		)
		num_batches = data["train"].dataloader.num_batches
		if args.max_steps is not None:
			args.max_epochs = ceil(args.max_steps / num_batches)
		else:
			assert args.max_epochs is not None and args.max_epochs > 0
			args.max_steps = num_batches * args.max_epochs
		total_steps = args.max_steps
		scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

	scaler = GradScaler() if args.precision == "amp" else None

	# Log and save hyper-params.
	params_file = os.path.join(args.logs, args.name, "params_{}.txt".format(time_suffix))
	with open(params_file, "w") as f:
		for name in sorted(vars(args)):
			val = getattr(args, name)
			f.write(f"{name}: {val}\n")

	for name in sorted(vars(args)):
		val = getattr(args, name)
		logging.info(f"  {name}: {val}")

	logging.info("training")
	# Optionally resume from a checkpoint
	start_epoch = 0
	steps = 0
	# Automatically restore latest checkpoint if exists
	if args.resume is None:
		latest_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
		if os.path.isfile(latest_path):
			args.resume = latest_path
	if args.resume is not None:
		if os.path.isfile(args.resume):
			logging.info(f"=> begin to load checkpoint '{args.resume}'")

		checkpoint = torch.load(args.resume, map_location="cpu")
		sd = checkpoint["state_dict"]
		new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
		model.load_state_dict(new_sd)

		if args.reset_data_offset:  ## 接着训练
			start_epoch = checkpoint["epoch"] - 1
			steps = checkpoint["step"]
			data = get_data(args, epoch_id=start_epoch, max_txt_length=args.context_length)

		# Restore the optim state
		if not args.reset_optimizer and optimizer is not None:
			optimizer.load_state_dict(checkpoint["optimizer"])
			logging.info("=> optimizer state is restored from the checkpoint")
		logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {steps} steps)")

	else:
		logging.info("=> no checkpoint found at '{}'".format(args.resume))


	cudnn.benchmark = True
	cudnn.deterministic = False

	# save result
	num_patence = 3
	epochs_early_stopping = EarlyStopping(patience=num_patence, verbose=args.debug)
	args.should_save = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none')
	for epoch in range(start_epoch, args.max_epochs):
		logging.info(f'Start epoch {epoch + 1}')
		num_steps_this_epoch, epoch_train_res = train(model, data, epoch, optimizer, scaler, scheduler, args, steps)
		steps += num_steps_this_epoch
		if args.val_data is not None and args.valid_epoch_interval is not None and (
				(epoch + 1) % args.valid_epoch_interval) == 0:
			assert "val" in data, "Error: Valid dataset has not been built."

			# epoch 验证
			save_path = os.path.join(args.checkpoint_path, "epoch_best.pt")
			epoch_eval_res=  evaluate(model, data, epoch, args, steps)
			epoch_valid_loss = float(epoch_eval_res['loss'])
			epochs_early_stopping(epoch_valid_loss, args, epoch + 1, steps, model, optimizer, save_path)

			# 最后一轮或提前终止：
			if epochs_early_stopping.early_stop or epoch == args.max_epochs - 1:

				loss_json = json.dumps({"train": TRAIN_LOG, "valid": VALID_LOG}, indent=True)
				loss_save = os.path.join(args.checkpoint_path, 'loss_monitor.json')
				print(loss_json, file= open(loss_save, 'w'))

				# monitor(loss_save, args.checkpoint_path)
				save_path = os.path.join(args.checkpoint_path, "epoch_latest.pt")
				save_weights(args.name, epoch + 1, steps, model, optimizer, save_path)
				t1 = time.time()
				logging.info(
					"Saved checkpoint {} (epoch {} @ {} steps) epoch_valid_loss: {} (writing took {} seconds)".format(save_path, epoch + 1,
					                                                                             steps,
					                                                                             epoch_valid_loss,
					                                                                             time.time() - t1))
				if epochs_early_stopping.early_stop:
					logging.info("Early stopping: {} (epoch {} @ {} steps) epoch_valid_loss: {} (writing took {} seconds)".format(save_path, epoch + 1,
					                                                                             steps,
					                                                                            epoch_valid_loss,
					                                                                           time.time() - t1))

					break

			if epoch + 1 < args.max_epochs:
				data = get_data(args, epoch_id=epoch + 1, max_txt_length=args.context_length)


			# Saving latest checkpoints.
			if args.should_save and num_steps_this_epoch > 0:
				if (epoch + 1) == args.max_epochs or \
						(args.save_epoch_frequency > 0 and ((epoch + 1) % args.save_epoch_frequency) == 0 ):

					t1 = time.time()
					save_path = os.path.join(args.checkpoint_path, "epoch_best.pt")
					save_weights(args.name, epoch + 1, steps, model, optimizer, save_path)
					logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path,  epoch + 1, steps, time.time() - t1))



# Restore the optim state

if __name__ == '__main__':
    main()


