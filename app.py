#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FireRedASR2S WebUI
基于 Gradio 的语音识别系统 Web 界面
"""

import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import torchaudio
from textgrid import TextGrid, IntervalTier

from fireredasr2s.fireredasr2 import FireRedAsr2Config
from fireredasr2s.fireredlid import FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVadConfig
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
)
logger = logging.getLogger("fireredasr2s.webui")


# ==================== 工具函数 ====================

def write_textgrid(tg_dir, name, wav_dur, sentences, words=None):
    """生成 TextGrid 文件"""
    os.makedirs(tg_dir, exist_ok=True)
    textgrid_file = os.path.join(tg_dir, name + ".TextGrid")
    logger.info(f"Write {textgrid_file}")
    textgrid = TextGrid(maxTime=wav_dur)

    tier = IntervalTier(name="sentence", maxTime=wav_dur)
    for sentence in sentences:
        start_s = sentence["start_ms"] / 1000.0
        end_s = sentence["end_ms"] / 1000.0
        text = sentence["text"]
        confi = sentence["asr_confidence"]
        if start_s == end_s:
            logger.info(f"(sent) Write TG, skip start=end {start_s} {text}")
            continue
        start_s = max(start_s, 0)
        end_s = min(end_s, wav_dur)
        tier.add(minTime=start_s, maxTime=end_s, mark=f"{text}\n{confi}")
    textgrid.append(tier)

    if words:
        tier = IntervalTier(name="token", maxTime=wav_dur)
        for word in words:
            start_s = word["start_ms"] / 1000.0
            end_s = word["end_ms"] / 1000.0
            text = word["text"]
            if start_s == end_s:
                logger.info(f"(word) Write TG, skip start=end {start_s} {text}")
                continue
            start_s = max(start_s, 0)
            end_s = min(end_s, wav_dur)
            tier.add(minTime=start_s, maxTime=end_s, mark=text)
        textgrid.append(tier)
    textgrid.write(textgrid_file)
    return textgrid_file


def write_srt(srt_dir, name, sentences):
    """生成 SRT 字幕文件"""
    def _ms2srt_time(ms):
        h = ms // 1000 // 3600
        m = (ms // 1000 % 3600) // 60
        s = (ms // 1000 % 3600) % 60
        ms = (ms % 1000)
        r = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        return r

    os.makedirs(srt_dir, exist_ok=True)
    srt_file = os.path.join(srt_dir, name + ".srt")
    logger.info(f"Write {srt_file}")

    i = 0
    with open(srt_file, "w") as fout:
        for sentence in sentences:
            start_ms = sentence["start_ms"]
            end_ms = sentence["end_ms"]
            text = sentence["text"]
            if text.strip() == "":
                continue

            i += 1
            fout.write(f"{i}\n")
            s = _ms2srt_time(start_ms)
            e = _ms2srt_time(end_ms)
            fout.write(f"{s} --> {e}\n")
            fout.write(f"{text}\n")
            if i != len(sentences):
                fout.write("\n")
    return srt_file


# ==================== 全局模型缓存 ====================

class ModelCache:
    """模型缓存，避免重复加载"""
    def __init__(self):
        self.asr_system = None
        self.current_config_hash = None

    def get_config_hash(self, enable_vad, enable_lid, enable_punc, asr_type, use_gpu):
        """生成配置哈希，用于判断是否需要重新加载模型"""
        return f"{enable_vad}_{enable_lid}_{enable_punc}_{asr_type}_{use_gpu}"

    def get_or_create_system(self, config, progress=None):
        """获取或创建 ASR 系统"""
        config_hash = self.get_config_hash(
            config.enable_vad, config.enable_lid, config.enable_punc,
            config.asr_type, config.asr_config.use_gpu
        )

        if self.asr_system is None or self.current_config_hash != config_hash:
            if progress:
                progress(0.1, desc="正在加载模型...")
            logger.info(f"加载模型配置: {config_hash}")
            self.asr_system = FireRedAsr2System(config)
            self.current_config_hash = config_hash
            if progress:
                progress(0.3, desc="模型加载完成")

        return self.asr_system


model_cache = ModelCache()


def normalize_audio_to_16k_mono(audio_path, output_dir, progress=None):
    """将输入音频规范化为 16kHz mono wav，返回可用于推理的文件路径"""
    import soundfile as sf
    audio_info = sf.info(audio_path)
    sample_rate = audio_info.samplerate
    num_channels = audio_info.channels

    if sample_rate == 16000 and num_channels == 1:
        return audio_path

    if progress:
        progress(0.35, desc="音频重采样到 16kHz mono...")

    waveform, sr = torchaudio.load(audio_path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    normalized_path = os.path.join(output_dir, "normalized_16k_mono.wav")
    torchaudio.save(
        normalized_path,
        waveform,
        16000,
        encoding="PCM_S",
        bits_per_sample=16,
    )

    return normalized_path


# ==================== 推理函数 ====================

def process_audio(
    audio_file,
    enable_vad,
    enable_lid,
    enable_punc,
    asr_type,
    beam_size,
    asr_batch_size,
    punc_batch_size,
    use_gpu,
    progress=gr.Progress()
):
    """处理音频文件"""
    try:
        if audio_file is None:
            return "❌ 请上传音频文件", "", {}, None, None

        progress(0, desc="开始处理...")

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"outputs/outputs_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # 构建配置
        progress(0.05, desc="构建配置...")

        # 根据 ASR 类型选择模型路径
        if asr_type == "AED":
            asr_model_dir = "models/FireRedASR2-AED"
        else:
            asr_model_dir = "models/FireRedASR2-LLM"

        vad_config = FireRedVadConfig(use_gpu=use_gpu)
        lid_config = FireRedLidConfig(use_gpu=use_gpu)
        asr_config = FireRedAsr2Config(
            use_gpu=use_gpu,
            beam_size=beam_size,
            return_timestamp=True
        )
        punc_config = FireRedPuncConfig(use_gpu=use_gpu)

        system_config = FireRedAsr2SystemConfig(
            vad_model_dir="models/FireRedVAD/VAD",
            lid_model_dir="models/FireRedLID",
            asr_type=asr_type.lower(),
            asr_model_dir=asr_model_dir,
            punc_model_dir="models/FireRedPunc",
            vad_config=vad_config,
            lid_config=lid_config,
            asr_config=asr_config,
            punc_config=punc_config,
            asr_batch_size=asr_batch_size,
            punc_batch_size=punc_batch_size,
            enable_vad=enable_vad,
            enable_lid=enable_lid,
            enable_punc=enable_punc
        )

        # 获取或创建 ASR 系统
        asr_system = model_cache.get_or_create_system(system_config, progress)

        # 处理音频
        progress(0.4, desc="正在识别...")
        raw_audio_path = audio_file
        uttid = Path(raw_audio_path).stem
        audio_path = normalize_audio_to_16k_mono(raw_audio_path, output_dir, progress)
        result = asr_system.process(audio_path, uttid)

        progress(0.7, desc="生成输出文件...")

        # 提取结果
        text = result["text"]
        sentences = result["sentences"]
        words = result["words"]
        dur_s = result["dur_s"]

        # 生成语言识别结果摘要
        if enable_lid and sentences:
            lang_info = {}
            for sent in sentences:
                lang = sent.get("lang", "unknown")
                if lang:
                    lang_info[lang] = lang_info.get(lang, 0) + 1
            lang_summary = "\n".join([f"{lang}: {count} 句" for lang, count in lang_info.items()])
        else:
            lang_summary = "未启用语言识别"

        # 生成 TextGrid
        progress(0.8, desc="生成 TextGrid...")
        tg_file = write_textgrid(output_dir, uttid, dur_s, sentences, words)

        # 生成 SRT
        progress(0.9, desc="生成 SRT...")
        srt_file = write_srt(output_dir, uttid, sentences)

        # 保存完整结果
        result_json_path = os.path.join(output_dir, "result.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        progress(1.0, desc="完成！")

        return text, lang_summary, result, tg_file, srt_file

    except Exception as e:
        error_msg = f"❌ 处理失败：{str(e)}\n\n详细错误：\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg, "", {}, None, None


# ==================== Gradio 界面 ====================

def create_ui():
    """创建 Gradio 界面"""

    # 自定义 CSS
    custom_css = """
    .header-container {
        background: linear-gradient(135deg, #6B46C1 0%, #3B82F6 100%);
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    .header-title {
        color: white;
        font-size: 36px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-subtitle {
        color: white;
        font-size: 14px;
        margin-top: 15px;
        line-height: 1.6;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .main-container {
        max-width: 90%;
        margin: 0 auto;
    }
    .warning-text {
        color: #f59e0b;
        font-size: 13px;
        margin-top: 8px;
    }
    """

    with gr.Blocks(css=custom_css, title="FireRedASR2S WebUI") as demo:
        # 标题区域
        with gr.Column(elem_classes="main-container"):
            gr.HTML("""
                <div class="header-container">
                    <h1 class="header-title">FireRedASR2S 语音识别系统</h1>
                </div>
            """)

            # 主体区域
            with gr.Row():
                # 左侧：输入控制
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 音频输入")
                    audio_input = gr.Audio(
                        label="上传音频文件",
                        type="filepath",
                        sources=["upload"]
                    )
                    gr.HTML("""
                        <div class="warning-text">
                            ⚠️ 仅支持 16kHz 16-bit mono PCM wav 格式<br>
                            如需转换：ffmpeg -i input.wav -ar 16000 -ac 1 -acodec pcm_s16le output.wav
                        </div>
                    """)

                    gr.Markdown("### ⚙️ 模块配置")
                    enable_vad = gr.Checkbox(label="启用 VAD（语音活动检测）", value=True)
                    enable_lid = gr.Checkbox(label="启用 LID（语言识别）", value=False)
                    enable_punc = gr.Checkbox(label="启用 Punc（标点预测）", value=True)

                    gr.Markdown("### 🎯 ASR 配置")
                    asr_type = gr.Radio(
                        choices=["AED", "LLM"],
                        value="AED",
                        label="ASR 类型"
                    )

                    with gr.Accordion("高级参数", open=False):
                        beam_size = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Beam Size"
                        )
                        asr_batch_size = gr.Number(
                            value=1,
                            label="ASR Batch Size",
                            precision=0
                        )
                        punc_batch_size = gr.Number(
                            value=1,
                            label="Punc Batch Size",
                            precision=0
                        )
                        use_gpu = gr.Checkbox(label="使用 GPU", value=True)

                    process_btn = gr.Button("🚀 开始识别", variant="primary", size="lg")

                # 右侧：输出展示
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 识别结果")
                    text_output = gr.Textbox(
                        label="识别文本",
                        lines=8,
                        max_lines=15
                    )

                    lang_output = gr.Textbox(
                        label="语言识别结果",
                        lines=3
                    )

                    json_output = gr.JSON(
                        label="完整结果（JSON）"
                    )

                    gr.Markdown("### 📥 文件下载")
                    with gr.Row():
                        tg_file = gr.File(label="TextGrid 文件")
                        srt_file = gr.File(label="SRT 字幕文件")

        # 绑定事件
        process_btn.click(
            fn=process_audio,
            inputs=[
                audio_input,
                enable_vad,
                enable_lid,
                enable_punc,
                asr_type,
                beam_size,
                asr_batch_size,
                punc_batch_size,
                use_gpu
            ],
            outputs=[
                text_output,
                lang_output,
                json_output,
                tg_file,
                srt_file
            ]
        )

    return demo


# ==================== 主函数 ====================

if __name__ == "__main__":
    logger.info("启动 FireRedASR2S WebUI...")
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=False,
        inbrowser=True,
    )

