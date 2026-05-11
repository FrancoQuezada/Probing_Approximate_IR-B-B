SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

CXX        = g++
STD        = -std=c++11

# --- CPLEX / CONCERT paths ---
CPLEXDIR      = /opt/ibm/ILOG/CPLEX_Studio2211/cplex
CONCERTDIR    = /opt/ibm/ILOG/CPLEX_Studio2211/concert
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CPLEXINCDIR   = $(CPLEXDIR)/include
CONCERTINCDIR = $(CONCERTDIR)/include

# --- Output folders ---
OBJDIR = obj
BINDIR = bin
EXEC   = main
TARGET = $(BINDIR)/$(EXEC)

# --- Build flags ---
WARN   = -w
OPT    = -O2
DBG    = -g

CPPFLAGS = -I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -MMD -MP
CXXFLAGS = $(STD) $(DBG) $(OPT) $(WARN)

LDFLAGS  = -L$(CPLEXLIBDIR) -L$(CONCERTLIBDIR)
# keep your CPLEX lib name choice as before:
CPLEXLIB = cplex$(dynamic:yes=2210)
LDLIBS   = -lconcert -lilocplex -l$(CPLEXLIB) -m64 -lm -lpthread -w

# --- Sources ---
SRCS = \
	Main.cpp \
	GlobalVariables.cpp \
	Fixing_props.cpp \
	MaBranchBound.cpp \
	ApproxBranchBound.cpp

OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)
DEPS = $(OBJS:.o=.d)

# =========================
# Targets
# =========================

.PHONY: all ini clean cleanall dirs

all: ini

ini: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

# Pattern rule for all .cpp -> .o
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Include auto-generated dependency files
-include $(DEPS)

clean:
	@rm -f $(OBJDIR)/*.o $(OBJDIR)/*.d

cleanall:
	@rm -rf $(OBJDIR) $(BINDIR)
