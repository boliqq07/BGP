# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:51:33 2020
@author: Administrator
"""

import re
import smtplib
import socket
import subprocess
import time
from email.mime.text import MIMEText

import requests


def get_host_ip():
    """
    查询本机ip地址
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def emails(mail_host, mail_user, mail_pass, sender, receivers):
    """从登录邮箱，并发送邮件到另一个邮箱"""
    # 设置服务器所需信息
    # 163邮箱服务器地址
    # mail_host = 'smtp.qq.com'
    # # 163用户名
    # mail_user = '986798607'
    # # 密码(部分邮箱为授权码)
    # mail_pass = 'xsysbklcoduvb'
    # # 邮件发送方邮箱地址
    # sender = '986798607@qq.com'
    # # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
    # receivers = ['b20170590@xs.ustb.edu.cn']

    # 设置email信息
    # 邮件内容设置

    receivers = [receivers, ]

    if all((mail_host, mail_user, mail_pass, sender, receivers[0])):

        message = MIMEText('content: %s' % get_host_ip(), 'plain', 'utf-8')
        # 邮件主题
        message['Subject'] = 'IP address:%s' % get_host_ip()
        # 发送方信息
        message['From'] = sender
        # 接受方信息
        message['To'] = receivers[0]

        connect(mail_host, mail_user, mail_pass, sender, receivers, message)

    else:
        pass


# 登录并发送邮件

def connect(mail_host, mail_user, mail_pass, sender, receivers, message):
    try:
        #    1
        smtpObj = smtplib.SMTP()
        # 连接到服务器
        smtpObj.connect(mail_host, 25)
        #    2
        smtpObj = smtplib.SMTP_SSL(mail_host)
        # 登录到服务器
        smtpObj.login(mail_user, mail_pass)
        # 发送
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        # 退出
        smtpObj.quit()
        print('email and ip success')
        return True
    except BaseException:
        print('email and ip error')  # 打印错误
        return False


def get_Local_ipv6_address():
    """
    This function will return your local machine's ipv6 address if it exits.
    If the local machine doesn't have a ipv6 address,then this function return None.
    This function use subprocess to execute command "ipconfig", then get the output
    and use regex to parse it ,trying to  find ipv6 address.
    """
    getIPV6_process = subprocess.Popen("ipconfig", stdout=subprocess.PIPE)
    output = (getIPV6_process.stdout.read())
    ipv6_pattern = '(([a-f0-9]{1,4}:){7}[a-f0-9]{1,4})'
    m = re.search(ipv6_pattern, str(output))
    if m is not None:
        return m.group()
    else:
        return None


def ustb_login(un, pw):
    try:
        s = requests.Session()
        log_url = "http://202.204.48.82/"

        ip6 = get_Local_ipv6_address()

        post_data = {
            'DDDDD': un,
            'upass': pw,
            'v6ip': ip6,
            '0MKKey': '123456789'
        }

        my_headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip',
            'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4'
        }
        s.post(url=log_url, data=post_data, headers=my_headers)
        r = s.get("http://www.baidu.com")
        if (r.url == "http://www.baidu.com/"):
            return True
        else:
            return False

    except (ConnectionError, BaseException):

        return False


def out_circle(s_time):
    is_login = False
    while is_login is False:
        print("Re-login %s......" % username)

        is_login = ustb_login(username, password)

        if is_login is True:
            print("%s logined in." % username)
            emails(mail_host, mail_user, mail_pass, sender, receivers)
            s_time = time.time()

        interval = 300  # if it is not login, check interval is one minutes
        time.sleep(interval)

    else:
        in_circle(s_time)


def in_circle(s_time):
    is_login = True

    while is_login:
        print("Pass")

        is_login = ustb_login(username, password)

        if time.time() - s_time > 86390 and is_login is True:
            emails(mail_host, mail_user, mail_pass, sender, receivers)
            s_time = time.time()

        interval = 3600  # if it is login, check interval is one hour
        time.sleep(interval)

    else:
        out_circle(s_time)


def main():
    print("Landing>>>")
    # 循环
    s_time = 0
    try:
        out_circle(s_time)
    except BaseException:
        out_circle(s_time)


if __name__ == '__main__':
    """
    # 安装
    # 1.安装python
    # 2.安装依赖包：pip install smtplib, socket,time, email
    # 3.将当前文件放到启动文件夹：C:\ProgramData\Microsoft\Windows\Start Menu\Programs\StartUp
    # 4.删除开机密码.
    # 5.更改下面的参数1，参数2。
    # 6.设置当前文件‘打开方式’为1中安装的 python.exe 程序
    """

    " *****参数1（登录，必填）***** "
    username = "b20170590"  # 双引号中填入学号
    password = "ss125800"  # 双引号中填入密码

    " **** 参数2（ip地址发送，可选）**** "  # 如需跳过，可以设置: mail_host = None
    mail_host = 'smtp.qq.com'  # 发送方邮箱主机,格式：smtp.xx.xx.com,
    mail_user = '986798607'  # 发送方邮箱名称,

    sender = '986798607@qq.com'  # 发送方邮箱地址
    mail_pass = 'xsysbklcoduvbfff'  # 发送方邮箱密码(部分邮箱为授权码！)

    receivers = 'b20170590@xs.ustb.edu.cn'  # 邮件接受方邮箱地址

    main()
