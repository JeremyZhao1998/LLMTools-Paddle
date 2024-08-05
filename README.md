# LLMTools-Paddle

Training large language models to use external tools based on PaddlePaddle and PaddleNLP.

2023百度松果基金项目：面向通用工具运用的大模型自动工具发现、理解与学习

## 研究目标和研究内容

大语言模型受到了广泛的关注和应用。在实际使用大型语言模型进行任务解决时，特定具体任务应当能够利用工具作为语言模型生成内容的补充和辅助。大语言模型对工具应用场景的理解，以及对工具的调用能力，成为影响其在具体任务上性能的关键因素。本项目旨在研究如何利用大语言模型的上下文学习能力，帮助大语言模型根据输入任务指令自动识别任务所需工具，调用工具解决问题，提升大型语言模型对通用工具的使用效率和解决问题的准确率。

## 研究方案和技术路线

首先，从利用大语言模型的上下文理解能力，设计prompt框架，使大模型自主地从大规模预训练语料数据库中筛选潜在工具调用语料，并向其中插入工具调用语句，构建工具调用数据集。其次，使用工具调用数据集对大语言模型进行轻量级微调，使之学习自动判别具体任务并调用工具。最后，对于微调后的大语言模型，构建工具调用框架，使之在没有人工设计的专门提示前提下，自主利用工具调用结果提升具体任务上的准确率。

### 工具调用数据集构建

对于大规模预训练语料数据库 $\mathcal{C}=\{x^1,x^2,\ldots,\ x^{\left|\mathcal{C}\right|}\}$ ，本项目利用大语言模型 $M$ 的上下文理解能力，从中筛选语料并转化为工具调用数据集 $\mathcal{C}^\ast=\{x^1,x^2,\ldots,x^{\left|\mathcal{C}^\ast\right|}\}$ 。具体步骤如下：（1）对于一条语料样本 $x=\{x_1,x_2,\ldots,x_n\}$ ，对于某一特定工具调用：

$$e\left(API\right)=\left[API\left(c\right)\right]$$

其中 $[$ 是工具调用起始token，$]$ 是工具调用终止token，$API$ 是工具调用名称，$c$ 是工具调用所传递的参数。本项目设计了prompt模板 $P(x)$ 用以输入大语言模型 $M$，鼓励模型为 $x$ 插入工具调用语句。记

$$p_i=p_M\left(\ [\ \middle| P\left(x\right),x_{1:i}\right)$$

为模型在套用prompt模板后，在生成序列 $x$ 的第 $i$ 个位置插入了工具调用起始token的概率（即模型输出 $P\left(x\right),x_{1:i}$ 后紧接着输出 $[$ 的概率）。在此，本项目取阈值 $\tau_s$，筛选 $p_i>\tau_s$ 的位置 $I=\{i\ |\ p_i>\tau_s\}$ 进行工具调用语句插入。（2）对于语料 $x$ 的工具调用插入位置集合 $I$，尝试在每个位置插入工具调用起始token后，令模型继续生成过程。由于大语言模型具有良好的上下文学习和生成能力，其能够根据prompt模板 $P\left(x\right)$ 进行工具调用，成功启用工具名称并传递相应参数，记为 $e\left(API\right)=\left[API\left(c\right)\right]$，当生成工具调用终止token $]$ 后停止生成过程。此时，启用外部工具执行该工具调用：

$$e\left(API,r\right)=\left[API\left(c\right)\ ->\ r\right]$$

其中 $r$ 是工具调用所返回的结果。对于插入位置集合 $I$ 中的每个位置 $i$，比对其调用工具，获取结果并插入该位置后，能否让大语言模型 $M$ 在语料 $x$ 上的损失降低：记以 $z$ 为前缀，语料 $x$ 前 $i$ 个token的生成损失为：

$$L_i\left(z\right)=-\sum_{j=1}^{n}logp_M\left(x_j\middle| z,x_{1:j-1}\right)$$

又记以工具调用为前缀的生成损失为 $L_i^+=L_i\left(e\left(API\right)\right)$，以工具调用且返回结果为前缀的生成损失为 $L_i^-=L_i\left(e\left(API,\ r\right)\right)$，而不加入任何工具调用的原本损失为 $L_i\left(\emptyset\right)$。本项目设置阈值 $\tau_f$，筛选调用工具且返回结果令损失函数降低的工具调用插入，即满足 $min\left(L_i^-,L_i\left(\emptyset\right)\right)-L_i^+>\tau_f$ 的位置 $i$，作为最终工具调用数据集的一条样本：

$$x^\ast=\{x_1,x_2,\ldots,x_i,\left[API\left(c\right)\right],x_{i+1},\ldots,x_n\}$$

在具体实现中，本项目为每种工具生成10,000条工具调用样本作为工具调用数据集 $\mathcal{C}^\ast$。

### 轻量级微调

构建工具调用数据集后，本项目使用这些语料对大语言模型进行微调。以（1）中的方式筛选的工具调用语料，其本身在大语言模型预训练语料之中，同时也保证了模型损失能够降低，因此使用这些语料对模型进行微调时不会影响模型本身的上下文理解和生成能力，同时能够赋予模型感知输入语句并调用相应工具的能力。在微调中，本项目采用大语言模型预训练过程相同的语言建模损失。在具体实现中，本项目使用了使用PaddleNLP套件的预训练工具，使用bf16精度，sharding stage2并行策略。

### 工具调用框架

大语言模型微调完成后，即具有了自主判断输入语句，进行任务拆解并自动调用工具的能力。本项目在此基础上为其设计了工具调用框架，赋予大语言模型任务拆解后的工具调度与决策能力。具体而言，在大语言模型生成过程中，如果检测到当前文本适用于调用某种工具，基于微调赋予的任务拆解和自动调用工具能力，模型将在适当位置生成工具调用起始token，并生成完整的工具调用名称和参数 $\left[API\left(c\right)\right]$，此时工具调用框架将中断模型的序列生成，执行工具调用，并将结果返回：$\left[API\left(c\right)\ ->\ r\right]$。执行工具并返回的结果将被返还给大语言模型，从中断位置继续其序列生成过程。在这一过程中，大语言模型将根据工具调用结果进一步修正其序列生成，最终达到提升具体任务准确性的目的。

## 项目成果

本项目基于飞桨框架建设了一套工具调用数据集生成prompt框架，生成了工具调用数据集，利用该数据集微调大模型，并为微调后的大模型建立工具调用框架，使大语言模型初步具备输入指令任务拆解、工具调度、决策和执行能力，提升多个具体任务评测集上的性能10%以上。

在实际实现中，本项目使用的基础大语言模型为Llama2-7B-chat，使用的基础预训练语料库为C4数据集，所支持的五种工具为：

* 计算器（python内置）
* 问答（外部大语言模型API，保留接口，可灵活更换各种语言模型API）
* 翻译（外部大语言模型API，保留接口，可灵活更换各种语言模型API ）
* 日历（python内置）
* 搜索引擎（外部搜索引擎API，保留接口，可灵活更换各种搜索API）

其中，考虑到项目实现成本和评测环境，本项目暂时使用的问答和翻译模型为Deepseek，使用的搜索引擎为Wikipedia。

对于每种工具调用效果的评测，本项目使用了ASDIV（算数评测）、SQUAD（问答评测）、MLQA（多语种问答评测）、DateSet（时间日期相关评测）、NQ（常识评测）等多种评测集，分别测试了计算器、问答、翻译、日历和搜索引擎的工具调用效率。如表1所示，使用本项目框架的大语言模型在各个评测集上至少取得10%的大幅性能提升。


| 评测集 | ASDIV | SQUAD | MLQA | DateSet | NQ |
| -------- | -------- | -------- | -------- | -------- | -------- | 
| 调用工具 | 计算器 | 问答 | 翻译 | 日历 | 搜索 |
| Llama2-7b-chat | 26.10 | 4.44 | 9.33 | 1.71 | 28.60 |
| 本项目 | 53.00 | 25.23 | 21.02 | 32.50 | 40.25 |

## 项目运行准备

### 环境安装

本项目使用飞桨框架，依赖paddlepaddle-gpu>=2.3.0，并使用了paddlenlp-develop。为确保飞桨框架顺利运行，请首先确保运行环境满足：

* python >= 3.8
* CUDA >= 11.8
* CUDNN >= 8.6

安装完整的项目依赖，仅需：

```
pip install -r requirements.txt
```

随后安装paddlenlp所需外部算子：

```
cd external_ops
python setup.py install
```

### 模型准备

本项目使用Llama2-7b-chat作为基础模型。在网络畅通的条件下，首次运行会由paddlenlp自动下载至``<home>/.paddlenlp/models/meta-llama/Llama-2-7b-chat/`` 。用户也可以手动将其下载至任意路径，随后修改 ``configs`` 目录下文件中的 ``<path/to/your/model>`` 路径。


### 数据准备

#### 训练数据

本项目采用了C4数据集（英文）的子集作为基础训练数据 $\mathcal{C}$。从 https://huggingface.co/datasets/allenai/c4 下载C4数据集（英文子集 ``c4/en``），放置于任意目录，随后使用 ``train_dataset/extract_subset.py`` 为每一外部工具提取其子集。例如数据根目录为 ``./data``，为 ``calculator`` 工具提取子集

```
python train_dataset/extract_subset.py --data_root ./data --subset_name calculator
```

该脚本会处理C4数据集的英文子集 ``c4/en``，并在数据根目录下生成用于训练计算器工具的子集 ``c4/calendar``。

#### 评测数据

本项目采用了 asdiv和svamp（https://github.com/arkilpatel/SVAMP ）算数评测集、lama问答评测集（https://github.com/facebookresearch/LAMA 包含了SQUAD、GoogleRE）、WebQS（https://huggingface.co/datasets/Stanford/web_questions ）和NQ（https://github.com/google-research-datasets/natural-questions ）知识评测集、MLQA多语言问答评测集（https://github.com/facebookresearch/MLQA ）和DateSet时间日期相关评测集，来评测五种工具调用解决问题的能力。从相应网站下载数据文件，放置在数据根目录下。

完整的数据根目录文件列表为：

```
<path/to/your/data/root>
    |_ c4
        |_ en
        |_ calculator
        |_ calendar
        |_ qa
        |_ search
        |_ translator
    |_ svamp
        |_ cv_asdiv-a
        |_ cv_svamp_augmented
    |_ lama
        |_ Google_RE
        |_ Squad
        |_ WebQS
            |_ test.json
        |_ NQ
            |_ NQ-open.efficientqa.test.1.1.jsonl
    |_ MLQA_V1
        |_ test
            |_ test-context-en-question-xx.json
```

### 工具API准备

本项目使用的问答和翻译工具都使用了外部大语言模型API，辅以相应的prompt模板。在具体实现中，本项目使用了OpenAI API模板，调用了Deepseek大模型。

用户需要在 ``tool_api_calls/translator_api.py`` 和 ``tool_api_calls/qa_api.py`` 中，将相对应的 API Key 和 url 替换为自己申请的实际内容，以获取外部大模型权限。

## 项目运行

### 工具调用数据集构建

以计算器工具为例，``configs/convert_calculator.sh`` 进行了该工具调用数据集的构建。修改其中的 ``<path/to/your/model>`` 和 ``<path/to/yout/data/root>`` 为实际的模型路径和数据根目录，该脚本将生成针对计算器的工具调用数据集 $\mathcal{C}^*$ 。

```
sh configs/convert_calculator.sh
```

在实际实现中，本项目默认为每种工具调用生成 10,000 条数据作为数据集。该脚本将同时生成符合paddlenlp预训练格式的预处理后的数据，放置在数据根目录。在这一过程完成后，数据根目录的结构为：


```
<path/to/your/data/root>
    |_ c4
        |_ en
        |_ calculator
        |_ calculator_star
            |_ star-x.json
            |_ merged.idx
            |_ merged.bin
            |_ tokenized
        |_ ...
    |_ ...
```

### 轻量级微调

本项目使用生成的工具调用数据集对大模型进行微调。本项目使用了paddlenlp的大模型套件，采用了原有的预训练损失。在 ``configs/train_calculator.json`` 中修改相应的配置，随后使用 ``configs/train.sh`` 启动训练：

```
sh configs/train.sh
```
训练后的模型文件将存储在 ``configs/train_calculator.json`` 定义的 ``<path/to/your/trained_model>`` 路径下。

### 工具调用框架与效果评测

本项目在评测过程中，对于经过微调，具有自动调用工具能力的模型，工具调用框架将在检测到模型调用起始token，接收工具调用名称和参数后，中断模型的序列生成，执行工具调用，并将结果返回：$\left[API\left(c\right)\ ->\ r\right]$。执行工具并返回的结果将被返还给大语言模型，从中断位置继续其序列生成过程。

在 ``configs/eval_calculator/`` 目录下，启动相应的文件来分别对未微调的原始模型 ``llama_xxx.sh`` 和微调后的工具调用模型 ``tool_xxx.sh`` 进行评测：

```
sh configs/eval_calculator/llama_asdiv.sh
sh configs/eval_calculator/tool_asdiv.sh
```

### 参考文献

```
@article{schick2024toolformer,
  title={Toolformer: Language models can teach themselves to use tools},
  author={Schick, Timo and Dwivedi-Yu, Jane and Dess{\`\i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Hambro, Eric and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

### 其他链接

百度飞桨星河AI Studio社区链接：
https://aistudio.baidu.com/projectdetail/8224119?sUid=819473&shared=1&ts=1722877360083
