import argparse
import smtplib
from email.mime.text import MIMEText


def sendMail(add_receiver_list=[],title="defaultTitle",content="finish !! "):
    # 设置服务器所需信息
    # 163邮箱服务器地址
    mail_host = 'smtp.126.com'
    # 163用户名
    mail_user = 'tj20010424'
    # 密码(部分邮箱为授权码)
    mail_pass = 'FLOEVIYUAQXRTHKS'
    # 邮件发送方邮箱地址
    sender = "tj20010424@126.com"
    # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
    receivers = ['1157507000@qq.com']
    receivers.extend(add_receiver_list)

    # 设置email信息
    # 邮件内容设置
    message = MIMEText(content, 'plain', 'utf-8')
    # 邮件主题
    message['Subject'] = title
    # 发送方信息
    message['From'] = sender
    # 接受方信息


    # 登录并发送邮件
    try:
        smtpObj = smtplib.SMTP()
        # 连接到服务器
        smtpObj.connect(mail_host, 25)
        # 登录到服务器
        smtpObj.login(mail_user, mail_pass)

        # 发送
        for address in receivers:
            message['To'] = address
            smtpObj.sendmail(sender, receivers, message.as_string())
        # 退出
        smtpObj.quit()
        print('success')
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tangjia_exp')
    parser.add_argument('--exp_name', default='exp_name_undifined', help='random seed')  # 输出结果保存路径
    args = parser.parse_args()
    sendMail(title=args.exp_name)