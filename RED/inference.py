if __name__ == '__main__':
    LC = True

    if LC:
        from model.model_LC import _NetG
    else:
        from model.model import _NetG

    