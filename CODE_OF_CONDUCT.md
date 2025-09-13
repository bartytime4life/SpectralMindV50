# 🌌 SpectraMind V50 — Code of Conduct

This project follows the **Contributor Covenant Code of Conduct v2.1**, extended with principles of **scientific reproducibility**, **mission-grade integrity**, and **competition fairness**.

> TL;DR
> Be respectful, honest, and reproducible. Protect the challenge’s integrity. No harassment, no shortcuts, no secrets, no data exfiltration.

---

## 🤝 Our Pledge

We pledge to make participation in this project a harassment-free, inclusive, and professional experience for everyone, regardless of age, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, sexual identity and orientation, or discipline. We are committed to:

* Respectful collaboration across **ML, astrophysics, and software engineering**.
* Maintaining **reproducibility and transparency** in all contributions.
* Protecting the fairness and credibility of the **NeurIPS 2025 Ariel Data Challenge**.
* Upholding **NASA-grade engineering standards** for safety, security, and auditability.

---

## 💡 Our Standards

### Examples of positive behavior

* Writing **clear, tested, and reproducible** code (Hydra overrides, DVC stages; no hardcoded paths).
* Respecting the **ADR** process for non-trivial changes and documenting decisions.
* Providing **constructive, respectful feedback** in PRs and reviews.
* Citing sources, prior work, and relevant literature; crediting others’ contributions.
* Helping others succeed in **local, CI, and Kaggle** environments.
* Ensuring models obey **scientific plausibility** (e.g., smooth, non-negative spectra; FGS1 priority as per challenge).

### Examples of unacceptable behavior

* Harassment, trolling, derogatory comments, personal or political attacks.
* Plagiarism or submitting others’ work as your own; misattribution.
* Inserting **insecure code**, secrets, backdoors, or **data exfiltration** logic.
* Subverting **reproducibility** (hidden configs, hardcoded paths, non-deterministic hacks).
* Breaking competition rules (e.g., collusion; leaking test labels; unauthorized sharing of data).
* Misrepresenting results or fabricating performance in publications or reports.
* Doxing or publishing others’ private information without explicit permission.

---

## 🛰️ Mission-Grade Extensions

To safeguard scientific integrity and competition fairness:

* **Reproducibility**

  * All experiments must be reproducible via **Hydra + DVC**; deterministic artifact paths; schemas validated prior to submission.
  * No network calls in Kaggle runtime code; read from `/kaggle/input/**`, write to `/kaggle/working/**`.

* **Fairness & Compliance**

  * Adhere to NeurIPS/Kaggle rules and dataset licenses.
  * No collusion, label leakage, or rule circumvention.

* **Security**

  * No secrets in repo or history; use CI/Kaggle secrets only.
  * Follow **[SECURITY.md](./SECURITY.md)** for vulnerability reporting and dependency hygiene.

* **Data Handling**

  * Protect any non-public data; avoid re-identification or exfiltration.
  * Respect confidentiality notices and competition T\&Cs.

---

## 🛡️ Enforcement Responsibilities

* **Maintainers** are responsible for clarifying and enforcing standards, and may take appropriate corrective actions in response to unacceptable behavior.
* **Contributors** must follow this Code, **[SECURITY.md](./SECURITY.md)**, and **[CONTRIBUTING.md](./CONTRIBUTING.md)**.

**Sanctions Ladder (examples):**

1. Private warning and education
2. Rejection/reversion of offending PRs/commits
3. Temporary or permanent ban from participation
4. Disclosure to challenge organizers if fairness rules may be impacted

Actions are proportionate to severity, pattern, and impact. Where appropriate, we will document decisions while protecting privacy.

---

## 📢 Reporting

If you experience or witness unacceptable behavior:

* Email: **[conduct@spectramind-v50.org](mailto:conduct@spectramind-v50.org)** (preferred)
* If your concern is a **security vulnerability**, please follow **[SECURITY.md](./SECURITY.md)** (private reporting; no public issues).

**Please include (if safe to do so):**

* What happened, when, and where (links, screenshots)
* Who was involved and whether you are comfortable naming them
* Any context that may help us assess impact or risk

**Response timeline**

* Acknowledgement within **72 hours**
* Initial assessment and next steps within **14 days**
* Final outcomes communicated when resolved (with reporter’s privacy protected)

---

## 🔁 Appeals

If you believe an enforcement action was made in error or without sufficient context, you may appeal by replying to our decision email with any additional information you wish us to consider. Appeals are reviewed by maintainers **not** directly involved in the original decision where feasible.

---

## 🔐 Privacy & Confidentiality

We will handle all reports **confidentially**. Information is shared strictly on a **need-to-know** basis to investigate and resolve incidents. Reporters may request anonymity. We may disclose limited information to challenge organizers or hosting platforms **only** if required to address material rule violations or safety concerns.

---

## 📎 Related Policies

* **Security Policy:** **[SECURITY.md](./SECURITY.md)**
* **Contributing Guide:** **[CONTRIBUTING.md](./CONTRIBUTING.md)**
* **Competition Rules & Licenses:** follow NeurIPS/Kaggle and dataset terms

---

## ⚖️ Attribution

This Code of Conduct is adapted from the **[Contributor Covenant](https://www.contributor-covenant.org/)** v2.1, with SpectraMind extensions for:

* **Mission-grade reproducibility**
* **Kaggle compliance**
* **Scientific and astrophysical integrity**

For answers to common questions about this code of conduct, see the **[FAQ](https://www.contributor-covenant.org/faq)**.

---

**Thank you** for helping keep SpectraMind V50 respectful, fair, and scientifically sound.
