## Hi there 👋

### LM Studio GitHub task

This repository includes a manually triggered GitHub Actions workflow that sends a prompt to an LM Studio server through its OpenAI-compatible API.

Because LM Studio normally runs on your own machine, use a **self-hosted GitHub Actions runner** on the same machine or network as LM Studio. GitHub-hosted runners cannot reach `localhost` on your computer.

#### Setup

1. In LM Studio, start the local server and load the model you want to use.
2. Confirm the LM Studio server is available at `http://127.0.0.1:1234/v1`, or note your custom URL.
3. Register and start a self-hosted GitHub Actions runner for this repository.
4. In GitHub, set the repository variable `LM_STUDIO_BASE_URL` if you use a custom endpoint.
5. Optionally set the repository secret `LM_STUDIO_API_KEY` if your LM Studio server requires an API key. If not set, the workflow uses `lm-studio`.
6. Open **Actions → LM Studio Task → Run workflow**, enter a prompt, and set the loaded model name.

The workflow lives at `.github/workflows/lm-studio-task.yml`, and the Python helper is in `scripts/lm_studio_task.py`.

<!--
**adamscarmccoy-boop/adamscarmccoy-boop** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
