
#ifndef NEURO_EXC_STATIC_H
#define NEURO_EXC_STATIC_H

// TODO Aggiungere tutti gli errori qui...

/*********************************************************/
// Define per enumerare i tipi di errore
/*********************************************************/
#define _NEURO_EXC_ENUM \
			enum type : size_t\
			{\
				pippo = 0,\
				pluto,\
				none\
			};

/*********************************************************/
// Define con le stringhe statiche delle descrizioni
/*********************************************************/
#define _NEURO_EXC_STR \
			inline static std::string _str[type::none] =\
			{\
				"PIPPO",\
				"PLUTO",\
			};

#endif