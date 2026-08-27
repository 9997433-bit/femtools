# femtools 三轮编排最终报告

## 产物

- 分支：`cursor/femtools-sota-d551`
- PR：https://github.com/9997433-bit/femtools/pull/1
- 定位：原创、求解器无关的结构动力学 CAE 框架，功能对标 FEMtools 产品族（非官方、不拷贝专有代码）

## 能力清单（已落地）

Framework（core/io/fea/script/cli/gui/viz）、Dynamics（FRF/谐波/CB/MBA/FBA/时域）、Pretest & Correlation（EFI/MAC/配对）、Model Updating、Optimization（尺寸/SIMP/DOE）、MPE（p-LSCF/FDD/LSCE）、RBPE。

## 验收

pytest 21 passed / 3 skipped；ruff 全绿；5 个 examples PASS；边界探针 5/5。

金标：轴向杆 ~1e-16；悬臂梁前三阶 ≤0.025%；HEX8 单层弯曲 98.6%；10% E 修正恢复 ~1e-7%；截断 FRF ~3–4% < 5%。

## 明确不做

NI 采集硬件、商业求解器二进制驱动、完整 GUI 对标商业桌面。
