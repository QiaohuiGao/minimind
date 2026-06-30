# 面试速记

> 由 `learning_system interview` 自动抽取各天笔记里标了 💼 面试版 的问答。
**共 2 条面试问答。**


## day06

### Q2. 知识蒸馏里 temperature（除以 T）为什么能软化分布？为什么还要乘 T²？（Layer 6）

> "蒸馏里 temperature 用来**软化 teacher 的输出分布**。softmax 靠 exp 指数放大 logits 差距，分布往往很尖、把次优 token 的信息淹没了；**把 logits 除以 T 等比缩小差距，分布变平**，那些'次优但相关'的 token 概率显现出来——这就是蒸馏要传的 dark knowledge，比 one-hot 硬标签信息量大。
> 至于**乘 T²**：因为除以 T 会让 KL 的梯度按 1/T² 缩小，乘回 T² 让它的梯度量级和硬标签 CE 保持可比，这样两个 loss 加权才平衡、调 T 时训练才稳定。
> 极端情况：T→1 是原始分布，T→0 退化成 one-hot，T→∞ 趋于均匀。"

**追问可能**：T 一般取多少？→ 1~4 常用（本项目 1.5）。T 太大分布太平、信息糊；太小接近硬标签、失去软化意义。

### Q3. 蒸馏 KL loss 里，为什么 student 用 log_softmax 防数值问题，teacher 用普通 softmax 却不用担心？（Layer 6）

> "`F.kl_div` 第一个参数要 log 概率、第二个要普通概率，所以 student 取 log、teacher 不取。至于数值稳定：KL 逐元素是 `p·log p − p·log q`。teacher 的 log 那项是 `p·log p`，**乘的是它自己**，p→0 时整项趋于 0，自动安全，而且 teacher 是 detach 的常量没有梯度；student 的 log 那项是 `−p·log q`，**乘的是 teacher 的 p 不是 q**，一旦 q 下溢就会变成巨大值还带坏梯度。所以只有 student 必须用 `log_softmax`（log-sum-exp 直接算 log，避开下溢）。"
> **追问**：log_softmax 为什么比 log(softmax) 稳？→ 它不显式算出会下溢的中间概率，直接用 logsumexp 得到 log 概率。
