#### Git版本管理  

**在Linux上安装Git**

```
sudo apt-get install git
```

**在Windows上安装Git**

在Windows上使用Git，可以从Git官网直接[下载安装程序](https://git-scm.com/downloads)，然后按默认选项安装即可。

安装完成后，在开始菜单里找到“Git”->“Git Bash”，蹦出一个类似命令行窗口的东西，就说明Git安装成功！

![install-git-on-windows](https://www.liaoxuefeng.com/files/attachments/919018718363424/0)

安装完成后，还需要最后一步设置，在命令行输入：

```
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```

因为Git是分布式版本控制系统，所以，每个机器都必须自报家门：你的名字和Email地址。你也许会担心，如果有人故意冒充别人怎么办？这个不必担心，首先我们相信大家都是善良无知的群众，其次，真的有冒充的也是有办法可查的。

注意`git config`命令的`--global`参数，用了这个参数，表示你这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址。

##### **创建版本库**

```
cd d
$ mkdir mygit
$ cd mygit
```

第二步，通过`git init`命令把这个目录变成Git可以管理的仓库：

```
$ git init
```



把一个文件放到Git仓库只需要两步。

第一步，用命令`git add`告诉Git，把文件添加到仓库

```
$ git add test.md
```

执行上面的命令，没有任何显示，这就对了，Unix的哲学是“没有消息就是好消息”，说明添加成功。

第二步，用命令`git commit`告诉Git，把文件提交到仓库：

```
$ git commit -m "wrote a readme file"	
```



##### 把文件添加到版本库

![image-20220110094938633](C:\Users\lws\AppData\Roaming\Typora\typora-user-images\image-20220110094938633.png)

#### 时光机穿梭

运行`git status`命令看看结果：

```
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

`git status`命令可以让我们时刻掌握仓库当前的状态，上面的命令输出告诉我们，`readme.txt`被修改过了，但还没有准备提交的修改。

虽然Git告诉我们`readme.txt`被修改了，但如果能看看具体修改了什么内容，自然是很好的。比如你休假两周从国外回来，第一天上班时，已经记不清上次怎么修改的`readme.txt`，所以，需要用`git diff`这个命令看看：

![image-20220110095835613](C:\Users\lws\AppData\Roaming\Typora\typora-user-images\image-20220110095835613.png)

#### 远程仓库

![image-20220110114126171](C:\Users\lws\AppData\Roaming\Typora\typora-user-images\image-20220110114126171.png)

![image-20220110120120148](C:\Users\lws\AppData\Roaming\Typora\typora-user-images\image-20220110120120148.png)

https://github.com/maxLWS/test01