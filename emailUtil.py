import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class MyEmail():
    def __init__(self):
        # 初始化
        self.smtp = smtplib.SMTP()

        # 连接邮箱服务器地址
        self.smtp.connect('smtp.126.com')

        # 加入主题和附件，邮件体
        self.email_body = MIMEMultipart('mixed')

        # 发件人地址及授权码
        self.email_from_username = 'tj20010424@126.com'
        self.email_from_password = 'GSPKIEXJAGGPQEQE'

        # 登录
        self.smtp.login(self.email_from_username, self.email_from_password)
    def generate_email_body(self, email_title, email_content,email_to_list=['1157507000@qq.com']):
        """
        组成邮件体
        :param email_to_list:收件人列表
        :param email_title:邮件标题
        :param email_content:邮件正文内容
        :param attchment_path:附件的路径
        :param files:附件文件名列表
        :return:
        """
        self.email_body['Subject'] = email_title
        self.email_body['From'] = self.email_from_username
        self.email_body['To'] = ",".join(email_to_list)

        text_plain = MIMEText(email_content, 'plain', 'utf-8')
        self.email_body.attach(text_plain)


        # 发送邮件
        # 注意：此处必须同时指定发件人与收件人，否则会当作垃圾邮件处理掉
        self.smtp.sendmail(self.email_from_username, email_to_list, self.email_body.as_string())

    def exit(self):
        """
        退出服务
        :return:
        """
        self.smtp.quit()

def send_email(title="您的程序运行完毕！！！",content=""):
    email = MyEmail()
    email.generate_email_body(email_title=title, email_content=content)
    email.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--name', type=str, default=' ',
                        help='备注')

    opt = parser.parse_args()
    send_email(title="您的程序 {} 运行完毕！！！".format(opt.name))