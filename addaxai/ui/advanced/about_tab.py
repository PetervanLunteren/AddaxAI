"""About tab content for AddaxAI advanced mode."""

import webbrowser
from functools import partial
from tkinter import END, INSERT, DISABLED

from addaxai.i18n import t, lang_idx as i18n_lang_idx


def write_about_tab(about_text_widget, hyperlink, text_font="TkDefaultFont", scroll=None):
    """Populate the about tab text widget with formatted content.

    Args:
        about_text_widget: tk.Text widget to write into
        hyperlink: HyperlinkManager instance bound to about_text_widget
        text_font: Font family name string
        scroll: Scrollbar widget to configure (optional)
    """
    text_line_number = 1

    # contact
    about_text_widget.insert(END, t('contact_header'))
    about_text_widget.insert(END, ["Please also help me to keep improving AddaxAI and let me know about any improvements, bugs, or new features so that I can keep it up-to-date. You can "
                           "contact me at ",
                           "Por favor, ayúdame también a seguir mejorando AddaxAI e infórmame de cualquier mejora, error o nueva función para que pueda mantenerlo actualizado. "
                           "Puedes ponerte en contacto conmigo en ",
                           "Merci de m'aider à améliorer AddaxAI et de me signaler toute amélioration, bogue ou nouvelle fonctionnalité afin que je puisse le maintenir à jour."
                           " Vous pouvez me contacter à l'adresse suivante: "][i18n_lang_idx()])
    about_text_widget.insert(INSERT, "peter@addaxdatascience.com", hyperlink.add(partial(webbrowser.open, "mailto:peter@addaxdatascience.com")))
    about_text_widget.insert(END, [" or raise an issue on the ", " o plantear un problema en ", " ou rapporter un incident sur la "][i18n_lang_idx()])
    about_text_widget.insert(INSERT, t('github_page'), hyperlink.add(partial(webbrowser.open, "https://github.com/PetervanLunteren/AddaxAI/issues")))
    about_text_widget.insert(END, ".\n\n")
    about_text_widget.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 1
    about_text_widget.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 2

    # addaxai citation
    about_text_widget.insert(END, t('citation_header'))
    about_text_widget.insert(END, ["If you used AddaxAI in your research, please use the following citations. The AddaxAI software was previously called 'EcoAssist'.\n",
                            "Si ha utilizado AddaxAI en su investigación, utilice la siguiente citas. AddaxAI se llamaba antes 'EcoAssist'.\n",
                            "Si vous avez utilisé AddaxAI dans vos recherches, veuillez utiliser les citations suivantes. Le logiciel AddaxAI s'appelait auparavant « EcoAssist ».\n"][i18n_lang_idx()])
    about_text_widget.insert(END, "- van Lunteren, P., (2023). AddaxAI: A no-code platform to train and deploy custom YOLOv5 object detection models. Journal of Open Source Software, 8(88), 5581, https://doi.org/10.21105/joss.05581")
    about_text_widget.insert(INSERT, "https://doi.org/10.21105/joss.05581", hyperlink.add(partial(webbrowser.open, "https://doi.org/10.21105/joss.05581")))
    about_text_widget.insert(END, ".\n")
    about_text_widget.insert(END, t('citation_plus_models'))
    about_text_widget.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 1
    about_text_widget.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 1
    about_text_widget.tag_add('citation', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 1
    about_text_widget.tag_add('citation', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 2

    # development credits
    about_text_widget.insert(END, t('development_header'))
    about_text_widget.insert(END, ["AddaxAI is developed by ",
                            "AddaxAI ha sido desarrollado por ",
                            "AddaxAI est développé par "][i18n_lang_idx()])
    about_text_widget.insert(INSERT, "Addax Data Science", hyperlink.add(partial(webbrowser.open, "https://addaxdatascience.com/")))
    about_text_widget.insert(END, [" in collaboration with ",
                            " en colaboración con ",
                            " en collaboration avec "][i18n_lang_idx()])
    about_text_widget.insert(INSERT, "Smart Parks", hyperlink.add(partial(webbrowser.open, "https://www.smartparks.org/")))
    about_text_widget.insert(END, ".\n\n")
    about_text_widget.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 1
    about_text_widget.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end'); text_line_number += 2

    # config about_text
    about_text_widget.pack(fill="both", expand=True)
    about_text_widget.configure(font=(text_font, 11, "bold"), state=DISABLED)
    if scroll is not None:
        scroll.configure(command=about_text_widget.yview)
