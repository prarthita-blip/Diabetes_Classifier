{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "SessionClosedException",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mSessionClosedException\u001b[0m                    Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-55a8ce6c1a80>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     41\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     42\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'__main__'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 43\u001b[1;33m     \u001b[0mbmi_app\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-1-55a8ce6c1a80>\u001b[0m in \u001b[0;36mbmi_app\u001b[1;34m()\u001b[0m\n\u001b[0;32m      8\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mbmi_app\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 10\u001b[1;33m     \u001b[0mname\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0minput\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Whats Your name\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     11\u001b[0m     \u001b[0mput_text\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Hi, welcome to our website'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mname\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     12\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\input.py\u001b[0m in \u001b[0;36minput\u001b[1;34m(label, type, validate, name, value, action, onchange, placeholder, required, readonly, datalist, help_text, **other_html_attrs)\u001b[0m\n\u001b[0;32m    241\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0md\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    242\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 243\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0msingle_input\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem_spec\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalid_func\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpreprocess_func\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0monchange_func\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    244\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    245\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\session\\__init__.py\u001b[0m in \u001b[0;36minner\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    270\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mto_coroutine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    271\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 272\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mrun_as_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    273\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    274\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0minner\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\utils.py\u001b[0m in \u001b[0;36mrun_as_function\u001b[1;34m(gen)\u001b[0m\n\u001b[0;32m    293\u001b[0m     \u001b[1;32mwhile\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    294\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 295\u001b[1;33m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgen\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mres\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    296\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mStopIteration\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    297\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\io_ctrl.py\u001b[0m in \u001b[0;36msingle_input\u001b[1;34m(item_spec, valid_func, preprocess_func, onchange_func)\u001b[0m\n\u001b[0;32m    235\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    236\u001b[0m     \u001b[0mspec\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlabel\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mlabel\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0minputs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mitem_spec\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 237\u001b[1;33m     data = yield input_control(spec=spec,\n\u001b[0m\u001b[0;32m    238\u001b[0m                                \u001b[0mpreprocess_funcs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m{\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mpreprocess_func\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    239\u001b[0m                                \u001b[0mitem_valid_funcs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m{\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mvalid_func\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\session\\__init__.py\u001b[0m in \u001b[0;36minner\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    270\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mto_coroutine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    271\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 272\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mrun_as_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    273\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    274\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0minner\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\utils.py\u001b[0m in \u001b[0;36mrun_as_function\u001b[1;34m(gen)\u001b[0m\n\u001b[0;32m    293\u001b[0m     \u001b[1;32mwhile\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    294\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 295\u001b[1;33m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgen\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mres\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    296\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mStopIteration\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    297\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\io_ctrl.py\u001b[0m in \u001b[0;36minput_control\u001b[1;34m(spec, preprocess_funcs, item_valid_funcs, onchange_funcs, form_valid_funcs)\u001b[0m\n\u001b[0;32m    255\u001b[0m     \u001b[0msend_msg\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'input_group'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mspec\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    256\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 257\u001b[1;33m     \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32myield\u001b[0m \u001b[0minput_event_handle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem_valid_funcs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mform_valid_funcs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpreprocess_funcs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0monchange_funcs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    258\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    259\u001b[0m     \u001b[0msend_msg\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'destroy_form'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\session\\__init__.py\u001b[0m in \u001b[0;36minner\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    270\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mto_coroutine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    271\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 272\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mrun_as_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    273\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    274\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0minner\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\utils.py\u001b[0m in \u001b[0;36mrun_as_function\u001b[1;34m(gen)\u001b[0m\n\u001b[0;32m    293\u001b[0m     \u001b[1;32mwhile\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    294\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 295\u001b[1;33m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgen\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mres\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    296\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mStopIteration\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    297\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\io_ctrl.py\u001b[0m in \u001b[0;36minput_event_handle\u001b[1;34m(item_valid_funcs, form_valid_funcs, preprocess_funcs, onchange_funcs)\u001b[0m\n\u001b[0;32m    309\u001b[0m     \"\"\"\n\u001b[0;32m    310\u001b[0m     \u001b[1;32mwhile\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 311\u001b[1;33m         \u001b[0mevent\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32myield\u001b[0m \u001b[0mnext_client_event\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    312\u001b[0m         \u001b[0mevent_name\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mevent_data\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mevent\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'event'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mevent\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'data'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    313\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mevent_name\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'input_event'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\session\\__init__.py\u001b[0m in \u001b[0;36minner\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    270\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mto_coroutine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    271\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 272\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mrun_as_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    273\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    274\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0minner\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\utils.py\u001b[0m in \u001b[0;36mrun_as_function\u001b[1;34m(gen)\u001b[0m\n\u001b[0;32m    293\u001b[0m     \u001b[1;32mwhile\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    294\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 295\u001b[1;33m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgen\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mres\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    296\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mStopIteration\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    297\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\session\\__init__.py\u001b[0m in \u001b[0;36mnext_client_event\u001b[1;34m()\u001b[0m\n\u001b[0;32m    277\u001b[0m \u001b[1;33m@\u001b[0m\u001b[0mchose_impl\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    278\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mnext_client_event\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 279\u001b[1;33m     \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32myield\u001b[0m \u001b[0mget_current_session\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnext_client_event\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    280\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mres\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    281\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pywebio\\session\\threadbased.py\u001b[0m in \u001b[0;36mnext_client_event\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    130\u001b[0m         \u001b[0mevent\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mevent_mq\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    131\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mevent\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 132\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mSessionClosedException\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    133\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mevent\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    134\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mSessionClosedException\u001b[0m: "
     ]
    }
   ],
   "source": [
    "from pywebio.input import *\n",
    "from pywebio.output import *\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "model = joblib.load('diabetes_classifier_model.pkl')\n",
    "scaler=joblib.load('Scaler.pkl')\n",
    "\n",
    "def bmi_app():\n",
    "    name=input(\"Whats Your name\")\n",
    "    put_text('Hi, welcome to our website', name)\n",
    "    \n",
    "    \n",
    "    preg=input('How many pregnancies did you have?(0 if you are male)', type=NUMBER)\n",
    "    glucose=input('What is your Fasting Glucose level?',  type=NUMBER)\n",
    "    bp=input('What is your Blood Pressure?', type=NUMBER)\n",
    "    age=input('What is your age?',type=NUMBER)\n",
    "    weight=input('What is your weight?(In Kilograms)',type=NUMBER)\n",
    "    height=input('What is your height?(In centimeters)',type=NUMBER)\n",
    "    history=select('Does any of your parent have diabetes?', ['YES', 'NO'])\n",
    "    if history=='YES':\n",
    "        history=1\n",
    "    else:\n",
    "        history=0\n",
    "        \n",
    "    bmi=weight/(height/100)**2\n",
    "    \n",
    "    prediction=model.predict(scaler.transform([[preg,glucose,bp,bmi,age,history]]))\n",
    "    \n",
    "    if prediction[0]==1:\n",
    "        put_text(\"You have risk of getting Type 2 Diabetes\")\n",
    "    else:\n",
    "        put_text(\"You do not have risk of getting Type 2 Diabetes\")\n",
    "        \n",
    "    \n",
    "    \n",
    "    \n",
    "        \n",
    "        \n",
    "\n",
    "    \n",
    "if __name__ == '__main__':\n",
    "    bmi_app()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
