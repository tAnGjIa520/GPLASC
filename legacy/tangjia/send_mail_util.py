"""Email notifications were removed from the maintained repository."""


def sendMail(*args, **kwargs):
    print("Legacy email notifications are disabled.")


if __name__ == "__main__":
    sendMail()
