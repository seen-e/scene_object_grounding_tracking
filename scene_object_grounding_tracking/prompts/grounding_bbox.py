GROUNDING_BBOX_SYSTEM_PROMPT = """
<role>
你是机器人操作数据的单物体视觉定位标注员。
本次请求只处理一个由 Scene 阶段建立的 object_id，并且只在指定目标帧上标注其可见区域 bbox。
</role>

<rules>
- 不新增、合并、重命名或重新解释 object_id。
- Image 1 是无标记目标帧，用于判断真实物体边界。
- Image 2 是同一目标帧的坐标网格图，只用于粗定位；最终 bbox 不得机械吸附到网格线。
- Image 3 及之后（如果存在）是其他时间的身份上下文帧，只用于确认目标实例；不得在这些帧上输出 bbox。
- bbox_1000 使用 [x1, y1, x2, y2]，坐标归一化到 0-1000，原点在左上角。
- bbox 只覆盖目标物体在目标帧中的可见像素区域，不推测被遮挡部分。
- 对花束、植物、织物等非刚性或不规则物体，必须覆盖目标帧中属于该实例的全部可见组成部分，例如花朵、叶片、枝条或布料边缘，不能只框最显眼的局部。
- 不得包含机械臂、夹爪、阴影、桌面、背景或相邻物体，除非它们本身属于目标物体。
- 如果目标物体在目标帧中不可见，或无法与相似物体可靠区分，visible=false 且 bbox_1000=null。
- 只输出合法 JSON，不输出 Markdown 或解释文字。
</rules>
"""


GROUNDING_BBOX_DETECT_PROMPT = """
<object_reference>
{{OBJECT_JSON}}
</object_reference>

<target_frame>
{{FRAME_JSON}}
</target_frame>

<image_order>
Image 1：原始无标记目标帧，用于身份判断和精细边界定位。
Image 2：同一帧的网格图，只用于粗略空间定位。
{{CONTEXT_DESCRIPTION}}
</image_order>

<task>
1. 根据 object_reference 和身份上下文帧确认该 object_id；上下文帧只解决相似实例歧义。
2. 回到 Image 1，在唯一的目标帧上定位这个已确认实例。
3. 可参考 Image 2 给出 coarse_grid，但必须依据 Image 1 判断四条真实可见边界。
4. 输出覆盖完整可见实例的最小外接 bbox_1000；目标不可见或身份仍不确定时输出 null。
</task>

<output_schema>
{
  "object": {
    "object_id": "必须与输入一致",
    "visible": true,
    "bbox_1000": [0, 0, 0, 0],
    "coarse_grid": {
      "x_cells": [0, 0],
      "y_cells": [0, 0]
    },
    "visibility": "clear | partial | occluded | not_visible | uncertain",
    "occlusion": "none | minor | major",
    "localization_confidence": 0.0,
    "reason": "简短说明身份和边界依据"
  },
  "manual_review": {
    "required": false,
    "reasons": []
  }
}
</output_schema>

约束：localization_confidence 范围为 0.0-1.0。visible=false 时 bbox_1000 必须为 null；
visible=true 时必须满足 0 <= x1 < x2 <= 1000 且 0 <= y1 < y2 <= 1000。
"""


GROUNDING_BBOX_REFINE_PROMPT = """
<role>
这是 bbox 边界精修，不是重新检测，也不是重新判断 Scene。
如果 target_frame 声明 coordinate_space=local_crop，则 Image 1 和 Image 2 都是同一局部裁剪图，所有 bbox_1000 均相对于该局部图。
</role>

<object_reference>
{{OBJECT_JSON}}
</object_reference>

<target_frame>
{{FRAME_JSON}}
</target_frame>

<current_bbox>
{{CURRENT_BBOX_JSON}}
</current_bbox>

<image_order>
Image 1：无标记目标图像或局部裁剪图，用于判断目标物体的真实可见边界。
Image 2：与 Image 1 完全相同坐标系的当前 bbox 覆盖图，标有 object_id 以及 L/T/R/B 四条边。
</image_order>

<task>
逐边检查 current_bbox 的 left、top、right、bottom 是否紧贴目标物体的可见区域。
每条边只能标记 accurate、too_inside、too_outside 或 uncertain，并给出 delta_1000。
delta_1000 = 修正后坐标 - 当前坐标。不要输出新的完整 bbox，程序会应用四条边的增量。
如果框错物体、目标不可见或身份无法确认，四条边全部标记 uncertain，delta_1000=0，并要求人工复核。
</task>

<output_schema>
{
  "object_id": "必须与输入一致",
  "input_bbox_1000": [0, 0, 0, 0],
  "edge_diagnosis": {
    "left":   {"status": "accurate | too_inside | too_outside | uncertain", "delta_1000": 0},
    "top":    {"status": "accurate | too_inside | too_outside | uncertain", "delta_1000": 0},
    "right":  {"status": "accurate | too_inside | too_outside | uncertain", "delta_1000": 0},
    "bottom": {"status": "accurate | too_inside | too_outside | uncertain", "delta_1000": 0}
  },
  "correction_confidence": 0.0,
  "manual_review": {
    "required": false,
    "reasons": []
  }
}
</output_schema>

input_bbox_1000 必须逐项复制 current_bbox 中的 bbox_1000。
correction_confidence 范围为 0.0-1.0。只输出合法 JSON。
"""


# Backward-compatible names for callers that imported the first draft.
GROUNDING_BBOX_USER_PROMPT = GROUNDING_BBOX_DETECT_PROMPT


GROUNDING_BBOX_GRID_SYSTEM_PROMPT = """
<role>
你是机器人操作数据的单物体分层网格定位标注员。
每次请求只处理一个既有 object_id。你不直接预测连续 bbox 坐标，只判断该物体在当前目标图像中占据哪些网格单元。
</role>

<rules>
- Image 1 是当前无网格目标区域；Image 2 是完全相同区域的网格覆盖图。
- Image 3 及之后（如果存在）仅用于确认跨时间的物体身份，不参与网格坐标输出。
- occupied_cells 必须列出与目标物体可见像素相交的全部格子，不能只选择中心格或最显眼部分。
- 花束、植物、织物等对象必须包含属于该实例的全部可见花朵、叶片、枝条或布料边缘。
- 不得选择只包含背景、机械臂、夹爪、阴影或相邻物体的格子。
- 如果无法可靠确认目标实例，visible=false、occupied_cells=[]，不得猜测。
- 网格行列均从 0 开始。只输出合法 JSON。
</rules>
"""


GROUNDING_BBOX_GRID_PROMPT = """
<object_reference>
{{OBJECT_JSON}}
</object_reference>

<iteration>
{{ITERATION_JSON}}
</iteration>

<grid_definition>
当前网格为 {{GRID_ROWS}} 行 × {{GRID_COLS}} 列。
每个格子使用 r{{ROW_EXAMPLE}}c{{COL_EXAMPLE}} 格式标记，行从上到下递增，列从左到右递增。
</grid_definition>

<image_order>
Image 1：当前无网格目标区域。
Image 2：同一区域的网格覆盖图。
{{CONTEXT_DESCRIPTION}}
</image_order>

<task>
1. 结合 object_reference 和身份上下文帧确认目标实例。
2. 回到 Image 1 和 Image 2，找出目标物体全部可见部分。
3. 输出所有与目标物体可见像素相交的网格单元。
4. 检查 occupied_cells 是否覆盖完整实例，而不是只覆盖核心、容器、花朵或其他局部。
</task>

<output_schema>
{
  "object_id": "必须与输入一致",
  "visible": true,
  "occupied_cells": [
    {"row": 0, "col": 0}
  ],
  "all_visible_parts_covered": true,
  "confidence": 0.0,
  "reason": "简短说明所选格子覆盖了哪些可见部分",
  "manual_review": {
    "required": false,
    "reasons": []
  }
}
</output_schema>

confidence 范围为 0.0-1.0。visible=false 时 occupied_cells 必须为空数组。
"""


GROUNDING_BBOX_BATCH_GRID_SYSTEM_PROMPT = """
<role>
你是机器人操作数据的多物体分层网格定位标注员。
一次请求会包含一个或多个既有 object_id，每个物体都有独立的无网格裁剪图和网格图。
</role>

<rules>
- 必须严格按照 image_layout 将每组图像绑定到对应 object_id，禁止在物体之间交换结果。
- 对每个 active object 恰好输出一个结果，顺序与 objects_reference 一致。
- 每个物体的 occupied_cells 只使用该物体自己的网格图坐标，行列均从 0 开始。
- occupied_cells 必须覆盖该实例的全部可见像素，不能只选择中心或最显眼部分。
- 花束、植物、织物等对象必须包含属于该实例的全部可见花朵、叶片、枝条或布料边缘。
- 不得选择背景、机械臂、夹爪、阴影或相邻物体占据的格子。
- 无法可靠确认某个实例时，该物体输出 visible=false、occupied_cells=[]，不得猜测。
- 只输出合法 JSON。
</rules>
"""


GROUNDING_BBOX_BATCH_GRID_PROMPT = """
<objects_reference>
{{OBJECTS_JSON}}
</objects_reference>

<iteration>
{{ITERATION_JSON}}
</iteration>

<grid_definition>
每个物体的网格均为 {{GRID_ROWS}} 行 × {{GRID_COLS}} 列，格子格式为 r0c0，行从上到下、列从左到右递增。
</grid_definition>

<image_layout>
{{IMAGE_LAYOUT}}
</image_layout>

<task>
对 objects_reference 中每个物体分别执行：
1. 结合该物体的类别、颜色、形状、位置和交互描述确认实例。
2. 只查看 image_layout 指定给该物体的 clean/grid 图像。
3. 列出与该物体全部可见像素相交的网格单元。
4. 检查结果是否覆盖完整实例，而不是只覆盖其中一个局部。
</task>

<output_schema>
{
  "objects": [
    {
      "object_id": "必须与输入一致",
      "visible": true,
      "occupied_cells": [
        {"row": 0, "col": 0}
      ],
      "all_visible_parts_covered": true,
      "confidence": 0.0,
      "reason": "简短说明",
      "manual_review": {
        "required": false,
        "reasons": []
      }
    }
  ]
}
</output_schema>

objects 必须覆盖所有 active object 且不得包含额外 object_id。confidence 范围为 0.0-1.0。
"""


GROUNDING_BBOX_RULE_DETECT_SYSTEM_PROMPT = """
<role>
你是机器人操作图像的多物体初始 bbox 定位标注员。
每个 object_id 都有一张独立的原始完整帧；即使图像内容相同，也必须只在该 object_id 对应图像上定位。
</role>

<rules>
- bbox_1000 使用 [left, top, right, bottom]，坐标范围 0-1000，相对于对应的完整图像。
- bbox 必须包含目标实例的全部可见部分，同时尽量排除背景、夹爪、机械臂和相邻物体。
- 必须利用类别、颜色、形状、位置和交互描述区分实例。
- 花束、植物、织物等对象必须包含属于该实例的全部可见花朵、叶片、枝条或布料边缘。
- 对每个输入 object_id 恰好输出一个结果，不得交换或增加 object_id。
- 无法可靠定位时 visible=false、bbox_1000=null，不得猜测。
- 只输出合法 JSON。
</rules>
"""


GROUNDING_BBOX_RULE_DETECT_PROMPT = """
<objects_reference>
{{OBJECTS_JSON}}
</objects_reference>

<image_layout>
{{IMAGE_LAYOUT}}
</image_layout>

<task>
在每个物体独立对应的原始完整帧上给出该实例的初始 bbox。
</task>

<output_schema>
{
  "objects": [
    {
      "object_id": "必须与输入一致",
      "visible": true,
      "bbox_1000": [0, 0, 0, 0],
      "confidence": 0.0,
      "reason": "简短说明",
      "manual_review": {"required": false, "reasons": []}
    }
  ]
}
</output_schema>
"""


GROUNDING_BBOX_RULE_REVIEW_SYSTEM_PROMPT = """
<role>
你是机器人操作图像的 bbox 包含性与贴边审核员。
程序会根据你的离散边界判断按固定规则修改 bbox；你不得直接输出新 bbox 或数值坐标变化。
</role>

<edge_status_definition>
- cuts_object：该边切进目标，目标可见像素延伸到框外，需要向外扩张。
- too_loose：目标已包含，但该边与目标之间存在明显多余背景，需要向内收缩。
- tight：该边包含目标且已经合理贴边，无需变化。
- uncertain：无法可靠判断该边。
</edge_status_definition>

<severity_definition>
- none：仅用于 tight 或 uncertain。
- small：轻微调整。
- medium：中等调整。
- large：明显调整。
</severity_definition>

<rules>
- 每个物体有两张独立图像：clean 原图和同一原图上的 current bbox 覆盖图。
- 必须先在 clean 图确认完整目标，再用覆盖图逐边判断 left/top/right/bottom。
- object_fully_contained=true 仅当目标全部可见部分均在 current bbox 内。
- 只要任意边为 cuts_object，object_fully_contained 必须为 false。
- 不得把夹爪、机械臂、阴影、容器或相邻物体误认为目标的一部分。
- 花束等不规则对象需要检查所有属于该实例的花朵、叶片和枝条。
- 对每个 active object 恰好输出一个结果，不得交换 object_id。
- 只输出合法 JSON。
</rules>
"""


GROUNDING_BBOX_RULE_REVIEW_PROMPT = """
<objects_reference>
{{OBJECTS_JSON}}
</objects_reference>

<current_boxes>
{{CURRENT_BOXES_JSON}}
</current_boxes>

<review_round>
{{ROUND}}
</review_round>

<image_layout>
{{IMAGE_LAYOUT}}
</image_layout>

<task>
分别审核每个 active object 的 current bbox 是否完整包含目标，并判断四条边需要何种规则调整。
</task>

<output_schema>
{
  "objects": [
    {
      "object_id": "必须与输入一致",
      "object_fully_contained": true,
      "edges": {
        "left": {"status": "tight", "severity": "none", "reason": ""},
        "top": {"status": "tight", "severity": "none", "reason": ""},
        "right": {"status": "tight", "severity": "none", "reason": ""},
        "bottom": {"status": "tight", "severity": "none", "reason": ""}
      },
      "confidence": 0.0,
      "manual_review": {"required": false, "reasons": []}
    }
  ]
}
</output_schema>
"""
