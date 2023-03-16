## Windows 安装
1. 安装 [Anaconda](https://www.anaconda.com/)
2. 安装 [Python 3.10.6](https://www.python.org/downloads/windows/), 并添加Python到配置环境"PATH"
3. 安装 [git](https://git-scm.com/download/win).
4. 安装环境对应的[PyTorch](https://pytorch.org/get-started/locally/) 命令如： ```conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia```
5. 下载 stable-diffusion-webui 代码, 如: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
6. 非管理员用户运行目录下的 `webui-user.bat`

## 错误处理
使用的命令皆需要在目录```stable-diffusion-webui\venv\Scripts\```的```python.exe```、```pip.exe```

**80%以上的失败，都是因为没科学上网**

1. 无法安装gfpgan
> 原因:是网络问题，就算已经科学上网，并设置为全局，也无法从github上下载源代码，从而导致install失败。

> 解决方法是直接到github下载 [GFPGAN](https://github.com/TencentARC/GFPGAN.git) 代码到本地，并进行本地安装。

> 因为stable diffusion会在其根目录创建虚拟python环境venv，因此安装方法与github有所不同。可参考以下方法：
- 从github将GFPGAN的源文件下载到本地，这一步可以使用git clone也可以直接下载zip文件。下载后，解压（如果用git clone就不需要）到d:\\stable-diffusion-webui\venv\Scripts目录下（stable-diffusion-webui是你stable diffusion webui的根目录，这个地址只是我电脑中的，请根据自己放的位置调整）。

- 打开cmd，cd到GFPGAN目录下，如：d:\\stable-diffusion-webui\venv\Scripts\GFPGAN-master下。

- 使用命令```d:\\stable-diffusion-webui\venv\Scripts\python.exe -m pip install basicsr facexlib```安装GFPGAN的依赖。

- 再使用```d:\\stable-diffusion-webui\venv\Scripts\python.exe -m pip install -r requirements.txt```安装GFPGAN的依赖。

- 使用```d:\\stable-diffusion-webui\venv\Scripts\python.exe setup.py develop```安装GFPGAN。


## UI本地化
选用[stable-diffusion-webui-localization-zh_CN](https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN.git)

### 1. 通过官方扩展列表安装
  此扩展可以在 **Extension** 选项卡里面通过加载官方插件列表直接安装
  - 点击 `Extension` 选项卡，点击 `Avaliable` 子选项卡
  - **取消勾选** `localization`，再把其他勾上，然后点击 **橙色按钮**，如下图
  ![image](https://user-images.githubusercontent.com/21131439/220507253-65b91219-05ac-4932-a129-0fcd1e55ffaa.png)

  - 在 `zh_CN Localization` 这一项的右边点击 `install`
  ![image](https://user-images.githubusercontent.com/21131439/220507520-77eab48a-272b-4a06-a38a-ca721181092f.png)
  - 安装完成，跳转到 [如何使用](#如何使用)

  ### 2. 或者，通过网址安装
  - 点击 `Extension` 选项卡，点击 `Install from URL` 子选项卡
  - 复制本 git 仓库网址：
  ```
  https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN
  ```
  - 粘贴进 URL 栏，点击 `Install`，如图
  ![image](https://user-images.githubusercontent.com/60730393/202898107-e207d645-e446-456c-8a5b-6dd400eba480.png)  
  - 安装完成，跳转到 [如何使用](#如何使用)

  ### 3. 又或者，直接下载然后放在对应路径
  - [下载本 git 仓库](https://codeload.github.com/dtlnor/stable-diffusion-webui-localization-zh_CN/zip/refs/heads/main)为 zip 档案
  ![image](https://user-images.githubusercontent.com/60730393/202898203-8f4265ff-efc1-4cb4-887a-86af291c000e.png)  

  - 解压，并把文件夹放置在 webui 根目录下的 `extensions` 文件夹中，放好之后应该会如下图
  ![image](https://user-images.githubusercontent.com/60730393/202898631-e4f6b3e2-b1d2-4258-b003-3142597fff3b.png)  
  - 安装完成，跳转到 [如何使用](#如何使用)



### 如何使用
  
  **确保扩展已经正确加载**  
  
  - 重启webUI以确保扩展已经加载了  
  
  - 在 `Settings` 选项卡，点击 **页面右上角**的 **橙色 `Reload UI` 按钮** 刷新扩展列表  
    ![image](https://user-images.githubusercontent.com/21131439/220509147-89b29802-2f9f-4db2-a21d-2dc99afa2d96.png)  

  - 在 `Extensions` 选项卡，确定已勾选本扩展☑️；如未勾选，勾选后点击**橙色按钮**启用本扩展。  
    ![image](https://user-images.githubusercontent.com/21131439/220509469-5c2af595-aece-4405-88f4-eb0638f8f22a.png)  

  **选择简体中文语言包（zh_CN）**  
  
  - 在 `Settings` 选项卡中，找到 `User interface` 子选项  
    ![image](https://user-images.githubusercontent.com/21131439/220509760-b8680fcd-9673-47e3-ba47-2ae0baf41d51.png)  
  
  - 然后去页面最底部，找到 `Localization (requires restart)` 小项，找到在下拉选单中选中 `zh_CN` （如果没有就按一下🔄按钮），如图  
  ![image](https://user-images.githubusercontent.com/21131439/220510690-4445c0bc-b70b-4943-b69c-270faa7cffc1.png)  

  - 然后按一下 页面顶部左边的 **橙色 `Apply settings` 按钮** 保存设置，再按 右边的 **橙色 `Reload UI` 按钮** 重启webUI  
  ![image](https://user-images.githubusercontent.com/21131439/220510486-90a1cf87-345b-48a7-8286-26dc02c0634e.png)  
  
  ## 修改配置
  ### 默认词
  - 文本方式打开目录下的```ui-config.json```文件
  - 编辑```"txt2img/Prompt/value"```、```"img2img/Prompt/value"```（正面词）的值
  - 编辑```"txt2img/Negative prompt/value"```、```"img2img/Negative prompt/value"```（负面词）的值
  - 保存，重启webUI  
  
  
