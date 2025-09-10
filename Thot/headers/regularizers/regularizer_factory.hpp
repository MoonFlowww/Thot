#pragma once

#include "regularizer.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Thot {
    namespace Regularizers {

        class Factory {
        public:
            using Creator = std::function<RegularizerPtr()>;

            static Factory& instance() {
                static Factory f;
                return f;
            }

            void register_creator(const std::string& name, Creator creator) {
                creators_[name] = std::move(creator);
            }

            RegularizerPtr create(const std::string& name) const {
                auto it = creators_.find(name);
                if (it == creators_.end()) {
                    throw std::runtime_error("Unknown regularizer: " + name);
                }
                return it->second();
            }

            std::vector<RegularizerPtr> create(const std::vector<std::string>& names) const {
                std::vector<RegularizerPtr> regs;
                regs.reserve(names.size());
                for (const auto& n : names) {
                    regs.push_back(create(n));
                }
                return regs;
            }

        private:
            std::unordered_map<std::string, Creator> creators_;
        };

        inline void register_regularizer(const std::string& name, Factory::Creator creator) {
            Factory::instance().register_creator(name, std::move(creator));
        }

        inline RegularizerPtr create_regularizer(const std::string& name) {
            return Factory::instance().create(name);
        }

        inline std::vector<RegularizerPtr> create_regularizers(const std::vector<std::string>& names) {
            return Factory::instance().create(names);
        }

    } // namespace Regularizers
} // namespace Thot
