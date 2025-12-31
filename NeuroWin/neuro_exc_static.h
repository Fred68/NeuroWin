
#ifndef NEURO_EXC_STATIC_H
#define NEURO_EXC_STATIC_H

/*********************************************************/
// Define per enumerare i tipi di errore
/*********************************************************/
#define _NEURO_EXC_ENUM \
			enum type : size_t\
			{\
				activation_function = 0,\
				EI_mismatch,\
				beta_mismatch,\
				learn_data_index,\
				init_data,\
				layer_number,\
				layer_out_of_range,\
				node_out_of_range,\
				size_mismatch,\
				pippo,\
				pluto,\
				none\
			};

/*********************************************************/
// Define con le stringhe statiche delle descrizioni
/*********************************************************/
#define _NEURO_EXC_STR \
			inline static std::string _str[type::none] =\
			{\
				"activation function type not recognized",\
				"ei is not set",\
				"beta is not set",\
				"learn data index is wrong",\
				"initialization data are invalid",\
				"layer number is wrong!",\
				"layer number is out of range",\
				"node number is out of range",\
				"size mismatch",\
				"PIPPO",\
				"PLUTO",\
			};

#endif