def all_none(centre_list):
    """
    :param centre_list:
    :return: checks if all positions in a list are None
    """
    none_yes = 1

    for element in centre_list:

        if element is None:
            none_yes = none_yes * 1

        else:
            none_yes = none_yes * 0

    return bool(none_yes)
